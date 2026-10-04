# goinfer Metal audit — 2026-09-30

**Tree:** goinfer `844700f8` (2026-09-30) + aikit `v1.51.0` (the `gpu/` Metal bindings goinfer's `metal/` calls through) +
MLX `9c3d355` (2026-09-30; MLX paths below are relative to `mlx/backend/metal/` unless they start with `mlx/`). The
working repo is at `cd7eb362` (v0.20.0 and later): seven commits after the audited snapshot, none of them under `metal/`,
`decoder/model.go`, `decoder/fitguard.go`, `decoder/fitplan.go`, `docs/benchmarks.md` or `docs/tasks/red-october.md`
(checked with `git log 844700f8..HEAD -- <those paths>`), so every `path:line` below holds in the working tree for those
files. Files outside that list were read at the snapshot only.

**Supersedes** `docs/audit-metal-2026-09-12.md` (`da1e461`) as the live Metal audit. That document stays as the record;
its finding IDs (M-/C-/G-/N-) are carried forward, one row each, in §12. This document numbers its own findings by area
(`A-` to `F-`, class letter, two digits: `B-P03` is area B, performance, third).

**Ask:** redo the Metal audit and borrow techniques from MLX: performance-led, with the question "what does MLX's source
teach goinfer's own kernels and runtime", then correctness, gates and stale claims. The owner decided against adopting MLX
as a dependency (it would break the cgo-free single static binary, fork the numerics and mean porting 39 model families
onto a different quantisation scheme), so the MLX lens here is technique-by-technique, not adoption.

**Method:** six independent reviewers by area (A prefill; B decode kernels; C runtime; D MoE, paging, DeltaNet, gpt-oss;
E batched decode, speculative verify, multi-slot KV, LoRA, sampling; F correctness, gates and claims). Each read the
record first (the Sep 12 audit's §0, §5 and §7, `docs/tasks/red-october.md` §2/§6 and R1–R19, the measurement records
and the autoresearch notes inside kernel comments) so that no recorded negative is re-proposed, then the area's Metal
code in full, then the MLX analog, and each carried forward its share of the Sep 12 IDs. Where a recorded negative's
premise looks stale the finding is filed as a REVISIT with the premise named and the cheapest probe. The consolidation
(§0–§3 and §10–§13) read all six reports in full, removed duplicates, and re-read the source behind the two Criticals,
three Majors (B-P01's `hd == 128` guard, C-P01's scale cache, E-P01's floor check) and the `--embed-int4` question
(E-X01, settled against the working repo's `internal/loadflags`, §3). The six area reports are Part II, edited only for
scratch path names and cross-references.

**Limitations:** static only. No Metal device, no Go toolchain; nothing was built, run or timed. Every number is one of
(rec) a figure the repo recorded, with doc and line; (cnt) counted from the code's shape, with the count shown; (mlx) MLX
source, file and line; (proj) a projection, carrying its band and basis. A projection is not a measurement and nothing
here should be quoted as one. MLX's `*_nax*` kernels (Metal 4 tensor ops, M5 only) are out of reach on the M1 Pro and
appear only as notes. `testdata/`, `scripts/`, `cmd/`, `internal/prequant`, `decoder/*_test.go`, `docs/completed/` and
most of `docs/tasks/` were not in the audited snapshot; findings that lean on them say so.

**Status, 2026-10-01.** The findings describe the tree at `844700f8`. Commits since then under `metal/`,
`decoder/fitguard.go`, `decoder/fitplan.go` and `docs/benchmarks.md` (`git log 844700f8..HEAD` on those paths):

- `de1c7f17` (2026-09-30) **fixed both Criticals.** A-C01 ≡ F-C01 (`-kv i8` prefill) and F-C02 (the exact prefill
  attention past 4096 keys) now decline batched prefill and name the reason, with device tests that fail when the guards
  are removed. No Critical is open.
- Since 2026-10-01 the program in `docs/tasks/task-metal-audit-2026-10.md` fixes findings on a branch; the Track 0
  status column below says which. So far: F-G03 (the decode attention kernels tested past 4096 keys against a float64
  reference), T0.3's F-C03, D-C01 and A-C02, T0.4's C-C01 (on the Metal side; the fit guards are unchanged), and
  F-G01 (a default test of the steel prefill attention kernel against a float64 reference), A-G01 (the
  fast-prefill floor), C-G01 (the load/Close leak test reads the device's own allocation total), D-G02 (the
  gated softmax layer's two kernels checked against the CPU), F-G04 (answered: where the verify gate went),
  E-G01 and F-G02 (the MC3 identity checks run by default on a generated fixture), and T0.5 (the stale comments and
  doc lines; two of its findings did not hold, see the Track 0 row).
- 2026-10-02, on the branch: **C-P01 done.** The night probe (T1.8) measured the scale cache at 1361.2 MB of 3004.6 MB
  of Go heap on M26 (0.453, parked by its rule); the owner promoted it on the absolute size. Both pagers now stage each
  expert's f16 scales from its WeightMat (the mapping), and the pread stage reads them from the file with the nibbles;
  the cache and `int4DirectBytes` are gone. Bit-identical by an exhaustive binary16 round-trip test. A three-arm A/B
  on M26 (old cache, copy, pread), with this row's own kill line (decode worse than −3%) as its ship bar, shipped the
  pread build: decode 1.061× the old cache's, 8 of 8 rounds faster, and the Go heap at token 32 3004.6 → 1553.2 MB.
  Records are in the program doc.
- `aff6f5e8` and `8e73aef5` (2026-10-01) put the 2026-09-30 peer-sweep cells into `docs/benchmarks.md`. That supplies the
  served K=3900 TTFT cell T1.11 asked for (LEVEL, 0.983: 4256.6 ms against Ollama's 4184.8, cell h) and replaces the stale
  rows A-D01 and B-D01 name; §0 quotes those rows as they stood at the snapshot. Still owed: the short-K prefill rows
  (A-D01) and the 0.5B decode cells at 2048 and 3900 keys (B-D01, T1.11).
- The rest are release and dependency bumps, test skips on memory-guard declines (`4223d381`) and R17's `-backend auto`
  (`5946e8f7`). None closes a finding. Every other finding stands as written.

Outside the findings: Metal re-quantizes every int8 body weight to int4 (G10 in `docs/tasks/task-gpu-paths-2026-09.md`,
labelled since M-25). Running those weights natively is planned in `docs/tasks/task-metal-int8-2026-10.md`.

The order §10's items run in, under the run-budget rules and this Mac's limits, is `docs/tasks/task-metal-audit-2026-10.md`.

**Reading order.** Part I (§0–§3) is the summary: the shape of it, the MLX technique ledger, the register of Critical and
Major findings, and the cross-area notes. Part II (§4–§9) is the six area reports, A to F. Part III (§10–§13) is the
program, the items closed earlier, the carry-forward of the Sep 12 IDs, and what could not be settled statically. A
figure marked (proj) is a projection with its band, not a measurement, and should not be quoted as one.

---

## 0. The shape of it

**Metal has moved from "behind Ollama on prefill, at the launch ceiling on decode" to "level with Ollama at the graded
shapes, and behind mlx-lm on decode".** After R16 and R19 the in-process `PrefillLast` at K=3900 is 4.32 s against
Ollama's served 4.23 s (rec `metal-prefill-attn-2026-09-27.md:122,125-126`, itself labelled a projection until a served
cell exists; the served cell measured 2026-09-30 reads LEVEL, 0.983, see the status note above); after R17, R18 and
R18b decode is ahead of Ollama in the six graded cells, 1.5B and 7B at depth 128, 2048 and 3900 (rec
`docs/benchmarks.md:47`). Against mlx-lm 0.31.3, in a cross-session comparison on a different quantisation scheme, so a
direction and not a ratio to quote, the same depth-128 cells read 0.82× (1.5B, 89.3 against
109.1 tok/s) and 0.75× (7B, 29.4 against 39.1) (rec `r12-mlx-row-2026-09-18.md:17-22`; B (f)9). **T1.14 (2026-10-02,
same session, main): 0.833× (1.5B) and 0.828× (7B)**; the 7B's cross-session 0.75× overstated the gap. Since Sep 12, 144
commits have touched `metal/`, which went from 14 non-test files, 6,796 lines and 69 kernels to 18 non-test files, 11,154 lines and 93 kernels
(plus 200 test files, 36,879 lines; cnt over the snapshot).

**MLX's quantized GEMV is not where the remaining decode gap lives.** Counted from source, MLX's `qdot` and goinfer's
`sa_rows_acc` both do about 3.8–3.9 ALU operations per weight, and goinfer issues 1.7–3.3× fewer load instructions per
weight; the two layouts cost the same 0.5625 bytes per weight; MLX has no prologue or epilogue fusion that goinfer lacks
(goinfer runs 12 dispatches per layer, MLX's graph keeps bias, rope, cache update and residual as separate ops); and
MLX's `sdpa_vector` is the per-key shape R17 replaced (B (e)). What MLX has that goinfer does not is mostly generality
(head dims, group sizes, a crossover that follows the kernel pair) and, for MoE and DeltaNet, device-side scheduling.
Those are the borrowable items, and §1 lists each MLX technique with its status.

**The largest open terms, by projected effect on a shipped shape** (every band below is a projection; §10 gives the
probe and the kill line for each):

1. *Decode attention coverage* (B-P01, B-P02, B-P03). R17's `attention_fa_blk` serves hd=128, group size 6 or 7, and
   1536 or more keys, and nothing else. The 0.5B (hd=64, where attention is 61% of its 3900-key token), every other
   group size, and every depth below 1536 run the older kernels, and the 1536 floor was measured on the kernel R17
   replaced. Bands: 5–20% of a 1.5B token at 1024–1535 keys; 1.3–1.85× on the 0.5B at 2048–3900 keys; 1.15–1.4× on
   ungraded group sizes at 2048+ keys.
2. *Short prompts.* The 64-token floor is a gate floor, not a speed floor: at K=32 the batched pass was 2.51× faster
   than sequential and was never gated (A-P02, rec). Separately the bit-identical 8-row step that MC3 already ships
   should do a 32-token prompt in about 74 ms against 305 ms sequential, with no fidelity gate needed (E-P01, proj). And
   passes of 64 rows or fewer pay a 64-row GEMM tile at 24–32 threadgroups (A-P01).
3. *Prefill that runs as decode.* Paged-MoE prefill is M paged decode tokens at 61 synchronous command buffers each
   (M26: 16.8–18.7 s TTFT for a 96-token prompt, rec; D-P01), and the Qwen3.5/3.6/3.8 hybrids have no batched prefill at
   all (D-B01; the dense batched-to-sequential ratio on this machine is about 10×, rec).
4. *The decode GEMV residue.* The down projection stays on the coal body at 84 and 111 GB/s against 132 and 166 for
   gate/up (B-P05, REVISIT of M-10); the lane mapping leaves half the last trip idle at K=1536 and K=3584 (B-P04); and
   the token round-trips through the host between kernels (C-B01, the one structural runtime technique MLX has and
   goinfer lacks). Summing the upper ends of the bands, which overlap and one of which needs a fidelity gate, gives
   a 5–21% cut in token time on the 1.5B at depth 128 and 2–14% on the 7B. Matching mlx-lm there would take cuts of 18%
   (0.82×) and 25% (0.75×) (T1.14 since: 0.833× and 0.828× same session, so about 17% on both), so these items do not close the gap alone; about 5.6% of the 7B token is head bytes from a
   chosen int8 pin (B (a)3).
5. *Memory.* The paged-expert f16 scale cache is a second, anonymous copy of scales the v15 `.giw` already aliases:
   about 1.43 GB of the M26's 2,967 MB footprint (C-P01, cnt).

**Correctness: no wrong-output path on a default configuration; two on shipped options, both fixed 2026-09-30
(`de1c7f17`).** At the snapshot `PrefillLast` had no int8-KV guard, so `-kv i8` wrote f16 K/V into an int8-sized cache
(A-C01, found independently as F-C01; one line in `prefillOK`). The exact prefill attention kernel keeps scores in
`threadgroup float sc[4096]` indexed by absolute key position, so it overran above 4096 keys on head-dim-256 families or
with `GOINFER_METAL_FUSED_ATTENTION=0`, and the fit guard auto-pins contexts in that range (F-C02). Both were Critical by
effect and non-default by condition; the brief's definition says "default path", so they are marked "(non-default
option)" and the reader can reclassify. I confirmed both in the source (§3). A third, conditional Major, still open: the
decoder's fit guards price f32 KV and know nothing of Metal's 32768 ceiling, so an auto-pin can inflate Metal's KV 3–7×
over the 4096 default or exceed the ceiling and move the whole forward to CPU (C-C01).

**Gates: the production kernels are less covered than the ones they replaced.** R19's `attention_prefill_steel` is the
prefill attention for every hd=128 model and has no assertion in any test that runs by default (F-G01); the MC3 step,
chunked-prefill and speculative-verify identity tests all need `GOINFER_METAL_MC3=1` and a real checkpoint, and the
chunk-invariance record was taken about 13 hours before steel's gate run (F-G02, E-G01); the context-ceiling test
asserts two constants (F-G03); and expert-major MoE prefill is default ON on a gate that ran on a dense 1.5B, with MoE
tests that are a tiny-fixture cosine bar (D-G01).

**One question resolved.** Reviewer E flagged a 2026-09-28 record saying the `--embed-int4` default makes the Metal
resident decline to CPU. The working repo settles it: `internal/loadflags/loadflags.go:130-139` defaults it on except
with `-backend metal`, where it stays off so the resident path survives; an explicit `--embed-int4` on Metal runs on the
CPU as asked. The default Metal path is intact (§3, note 1).

**Counts.** 82 finding rows across the six reports; 80 distinct findings after removing the A-C01/F-C01 duplicate
and E-X01 (resolved): 2 Critical (both non-default, both fixed 2026-09-30), 21 Major (16 performance or memory, 1
correctness, 4 gates), 57 Minor. 78 were open on 2026-10-01; two Minors, A-D01 and B-D01, are half settled (the status
note above). Eleven findings are REVISITs of a recorded negative or closure, each naming the premise it thinks went stale:
B-P03, B-P05, C-B02, C-B03, D-P02, D-P03, D-P04, D-B05, A-P03, and, for the batched step only, E-P05 (M-16) and E-P08
("fused argmax"). All 41 N-, 10 G-, 10 C- and 16 M- IDs of
the Sep 12 audit have a status in §12.

---

## 1. The MLX technique ledger

What MLX does, what goinfer does today, and the verdict. "Ported" means the shape is in the tree and the record says
so; the residue after a ported technique is named, because the brief was not to report a ported technique as missing.

| MLX technique (source) | goinfer today | Verdict |
|---|---|---|
| Quantized GEMV `qmv_fast`: 16 weights per lane, two simdgroups of four rows, taken when K%512==0 (`kernels/quantized.h:756-821`, `quantized.cpp:488-495`) | R18/R18b: MLX's masked half-staged form, R=4 rows per simdgroup, bit-identical because the group integer sums are exact (|sum| ≤ 60,960 < 2^24). ALU ops per weight counted at parity, 3.8–3.9 | **Ported.** Residue: lane mapping idles half the last trip at K=1536 and 3584, with a bit-identical fix (B-P04); threadgroup geometry untested (B-P06); the down projection is still on the coal body (B-P05) |
| Quantized GEMM `qmm_t`, fixed 32×32 tile; `qmm_splitk` targeting ~512 threadgroups (`quantized.cpp:1087-1090,1152-1215`) | R16: 64×64 tile, equal to MLX's dense steel 's' tile on M1 Pro (`matmul.cpp:166-170`), about 3.2× over the retired kernel (rec) | **Ported**, and larger than MLX's quantized tile. Left: shape-keyed tile (A-P01, bit-identical); split-K (A-B01, not recommended: reorders the reduction, an M-keyed split breaks chunk invariance); fragment-direct epilogue (A-B02, 1–3% of a pass) |
| Steel attention: bq32/bk16/wm4/wn1, function-constant specialisation (`kernels/steel/attn/`, `scaled_dot_product_attention.cpp:510-516`) | R19 `attention_prefill_steel`: same blocking, register O, exp2; K=3900 PrefillLast 8171→4316 ms (rec) | **Ported.** Left: function constants (a branch per block), hd=256 with wn=2 (A-B03, unmeasured), P kept in f32 (F-B01, fidelity only). MLX does not share K/V across GQA heads either (`steel_attention.h:97`) |
| `sdpa_vector` 1-pass and 2-pass decode attention (`kernels/sdpa_vector.h`, `scaled_dot_product_attention.cpp:776-806,1565`) | R17 `attention_fa_blk`: K loaded once per key for all G heads, 64 partitions per KV head (the count MLX uses) | **goinfer is ahead by shape.** Borrow only the generality: head-dim templating (B-P01), runtime group size (B-P02), and the lesson that a crossover belongs to the kernel pair being switched (B-P03) |
| Quantized MoE: `gather_qmv`, sorted `gather_qmm_rhs`, device offsets and tile scheduler, bm=16 (`kernels/quantized.h:2087-2209,2426-2574`, `gather_mm_offsets.metal`, `kernels/utils.h:503-542`) | Host-grouped expert-major prefill for non-paged generic MoE (5 dispatches per active expert, 64-row tile, one sync per layer); decode runs 3k dispatches per layer | **Borrow:** D-B02 (device-side scheduling, bm=16; bit-identical if R16's inner loop is reused), D-P03 (one dispatch over the k slots: same bytes, 20 fewer dispatches per layer). MLX has no expert pager, so D-P01 is goinfer's own design item |
| MXFP4 / NVFP4 (`kernels/fp_quantized.h`, `fp4.h`, `fp8.h`) | gpt-oss MXFP4 is decoded to f32 and requantised to int4 absmax/7 at load | **Borrow, owner decision:** D-B03, native E2M1×E8M0 keeps the checkpoint's values and saves 0.6 GB; loader, kernel, aikit and gate change together |
| `gated_delta_update`: T-loop kernel plus chunked C=8 WY form (`kernels/gated_delta_update.h:169-366`) | DeltaNet: 12 dispatches per layer per token; `delta_rule` is thread-per-row with the state read twice and written twice; hybrids prefill as M decode tokens | **Borrow:** D-B01 (T-loop kernel with batched conv and gates), D-B05 (lane-per-Dk layout; changes the summation order, so an owner decision) |
| Command-buffer policy: 20–50 ops per buffer by chip, async commit (`device.cpp:604-622`, `eval.cpp:29-69`) | One command buffer, one serial encoder, one commit and wait per token; encode-ahead already hides about half the host time (0.83→0.45 ms on the 1.5B, rec) | **Nothing to borrow in batching.** The async token chain is C-B01: 0.30–0.45 ms per token (proj), 2.2–3.4% of a 1.5B token; C-P02 is its sampled-decode sibling |
| Concurrent dispatch plus untracked buffers (`device.cpp:580`, `allocator.cpp:15-16`) | Serial encoder; M-16 measured untracked at 99.6–101.5% of tracked (rec) | **No gain on the dense chain** (11 strictly chained dispatches per layer). REVISIT only for the batched step, where per-row attention dispatches are independent siblings (E-P05); aikit does not expose the dispatch-type call, so a multi-row kernel is the cheaper route |
| Residency sets and the wired limit (`resident.cpp:221-237`, `allocator.cpp:100-105,254-262`) | A residency set for the paged-MoE slot pool only, attached per encoder; dense weights are NoCopy-aliased, file-backed | **REVISIT C-B02:** the bisect that said pinning costs in proportion to set size ran on paged MoE, never on dense decode with aliased weights; the 7B alias costs 1.4% decode (rec) |
| Spin fence, `MLX_METAL_FAST_SYNCH` (`fence.cpp:11-50`) | The recorded shared-event test measured an `MTLSharedEvent` handshake at about 0% gain on a dense 1.5B | **REVISIT C-B03:** a different mechanism from the one tested, and the paged boundary cost has fallen from the recorded ~14 ms (D-P02) |
| RMSNorm single-row kernel, x held in registers (`kernels/rms_norm.metal:13-80`) | `rmsnorm_quant`: 3 device passes, 12 barriers, pinned reduction tree | **Borrow, small:** B-P09, a register-cached variant with the same pinned tree, 0.3–2% of a 1.5B token. MLX's own reduction order differs and must not be copied as written |
| Kernel fusion | 12 dispatches per layer: norm+quant, QKV+bias, rope2, gate\|up, swiglu+quant, residual in the o and down epilogues | **Nothing to borrow.** MLX has none of these fusions |
| Quantized layout: three arrays at group 64 against weights plus f16 scales at group 32 | 0.5625 bytes per weight in both; same transaction counts | No layout lesson |
| 4-bit LM head in mlx-lm checkpoints (not verifiable in the library source) | Head pinned int8 by decision; Metal cannot load `--embed-int4` | B-N01, owner decision: head bytes are about 5.6% of the 7B token, the byte share of the MLX gap |
| `ArgPartition` through a full sort (`sort.cpp:346-348`) | One-thread `moe_route`, exact and tie-stable | MLX pays more; D-P04 (one-simdgroup route) is a goinfer item |
| Unretained command-buffer references (`device.cpp:323,559`) | Retained | Recorded negative holds (C) |
| `rope_single` | `rope2`, same shape, no table in either | Equal |
| NAX / Metal 4 tensor ops (`*_nax*`) | Not applicable on M1 Pro | Out of reach (M5); not findings |

---

## 2. Register of Critical and Major findings

Full entries are in Part II. "Bits" is the bit-identity class of the proposed change: *identical* (same bytes out),
*gated* (leaves the exact lane; needs the pooled fidelity gate or an owner decision), or *n/a*.

| ID | Sev | Claim | Effect (band is a projection unless marked rec/cnt) | Bits | First step |
|---|---|---|---|---|---|
| A-C01 ≡ F-C01 | Critical (non-default `-kv i8`) | `PrefillLast` never checks `r.kvI8`; `kv_store_f16` writes half K/V into an int8-sized cache | Wrong K/V for the whole prompt; out-of-bounds write once position ≥ ctxCap/2 | n/a | **Fixed 2026-09-30:** `!r.kvI8` in `prefillOK`, a named decline in `PrefillLast`/`PrefillPath`; `TestPrefill_declinesInt8KV` (red without the guard) |
| F-C02 | Critical (non-default: ctx > 4096 with hd > 128, or fused attention off) | Exact `attention_prefill` stores `sc[s]` by absolute key index in a 4096-float threadgroup array | Overrun above 4096 keys; reachable via the fit guard's auto-pin (5689 and 12109–29666 positions recorded) | n/a | **Fixed 2026-09-30:** `PrefillLast` declines when the exact kernel would see more than `prefillExactAttnMaxKeys` (4096) keys; `TestPrefill_exactAttentionDeclinesPast4096Keys` (red without it) and `TestPrefillExactAttnBound` (the constant matches the kernel's array). F-G03's test is not replaced yet |
| A-P01 | Major | ≤64-row passes run o, down and qkv at 24–32 threadgroups and a 64-row tile for ≤32 real rows | C ≤ 32: −15…−40 ms of a 96–105 ms pass; C 33–64: −5…−18 ms | identical | BM=32 / BN=32 pipelines by shape; arms in `TestMC3S0PrefillGEMMSmallM` |
| A-P02 | Major | The 64-token floor is a gate floor; K=32 batched was 2.51× faster (rec) and never gated | K=32 TTFT ~305→~120 ms (rec ratio); K=16 ~1.6× | gated | Pooled-gate cells at K=16/32/48 |
| A-P03 | Major if confirmed | Small-M prefill attention at depth is probably latency-bound; the table that says so predates R19 | 0 or −15…−30 ms per pass at depth 2048 | identical / gated | Rerun `TestMC5_passCost` on the current build (no code) |
| A-P04 | Major (owner call) | `HiddenLast` (/v1/embeddings) is one synchronous command buffer per token | Pipelined 1.05–1.5× (identical); batched ~10× at K=512 (gated, gives up exact embeddings) | identical or gated | Pipeline tokens 0..K−2 through `ForwardEmbNoLogitsPipe` |
| B-P01 | Major | hd=64 never reaches the flash-decode kernels; on the 0.5B attention is 61% of the 3900-key token (rec) | 0.5B token 1.3–1.85× at 2048–3900 keys | gated | Template `attention_fa_blk` on head dim; in-sequence attention at 2048/3900 |
| B-P02 | Major | `attention_fa_blk` is instantiated for G=6,7 only | 1.15–1.4× at 2048+ keys on ungraded group sizes | gated | g4/g5/g8 instantiations; per-head equality against g7 |
| B-P03 | Major (REVISIT) | `attnFADepthFloor` = 1536 was measured on `attention_fa`, not on the block kernel that now runs above it | 5–20% of a 1.5B token at 1024–1535 keys | gated (bits change at 256–1535) | Test-only floor override and an interleaved sweep 256..1535 |
| B-P04 | Major | SA-family lane mapping leaves the last trip half idle at K=1536 and K=3584 | 1.5B 3–7%, 7B 1–3% | identical form exists | Fixed-bytes K sweep, then a lane-balanced prototype |
| B-P05 | Major (REVISIT of M-10) | The down projection stays on the coal body: the slowest int4 GEMV (84 and 111 GB/s against 132 and 166) | 1.5B 0–9%, 7B 0–10% | gated; breaks decode/verify agreement unless MC3's down kernel moves too | Dispatch `gemv_w4a8_sa_resid_rows4` at the down shape in `TestR18InSequence` |
| C-P01 | Major | Paged-expert f16 scale cache duplicates scales the v15 `.giw` aliases | −1.43 GB of 2,967 MB (−48%, cnt); decode −6…0% | identical | `vmmap` attribution, then pread the scale span |
| C-C01 | Major (conditional) | Decoder fit guards ignore Metal's 32768 ceiling and its f16 KV | Metal KV 3–7× the default at an auto-pin; or the whole forward moves to CPU | n/a | Unit test with injected free RAM and a 131072-window config |
| D-P01 | Major | Paged-MoE prefill is M paged decode tokens; no expert-major through the pool | 3–15× TTFT at M=512; M26 96-token TTFT 16.8–18.7 s (rec) | gated | `moe_prefill_measure_test.go` paged arm at M=64/128/512 |
| D-B01 | Major | No batched DeltaNet prefill; hybrids prefill as M decode tokens | 3–8× TTFT on resident hybrids | gated | T-loop kernel plus batched conv and gates |
| D-B02 | Major | Host-grouped expert loop: 5 dispatches per active expert, per-layer sync, 64-row tile padding | 5–25% of MoE prefill | identical if R16's inner loop is reused | Count tile waste, then a device offsets kernel |
| D-P04 | Major | MoE router top-k runs on one GPU thread | 4–9% of a resident MoE token | identical indices | GPU timestamps around the (1,1) dispatch |
| D-B03 | Major | gpt-oss MXFP4 is requantised to int4 absmax/7 | 0.6 GB (cnt); fidelity cost unquantified | gated | CPU-side requantisation-error script on the real checkpoint |
| E-P01 | Major | Prompts under 64 tokens and short reuse suffixes run sequentially or in a ≥96 ms pass; the bit-identical 8-row step does them at ~2.3 ms per token | K=32 TTFT 305→~74 ms; 7B ~1.06 s→~0.28 s | identical | Drive `forwardMultiInto` over a 32-token prompt in 8-row pieces, compare bytes and time |
| D-G01 | Major (gate) | Expert-major prefill ships default ON on a dense-model gate; the MoE tests are a tiny-fixture cosine bar | n/a | n/a | Real-MoE paged-off fixture, per-layer cosine, router-flip count, mutation set |
| F-G01 | Major (gate) | The production hd=128 prefill kernel (R19 steel) has no default-runnable assertion | n/a | n/a | Float64-reference test in the shape of `metal/attn_fa_blk_test.go:147` |
| F-G02 | Major (gate) | MC3, chunked-prefill and spec-verify gates are env-gated on a real checkpoint; the chunk-invariance record predates steel | n/a | n/a | Re-run `TestMC5_prefillChunkInvariance` on the wired build; add a default-run fixture |
| F-G03 | Major (gate) | `TestMetalCtxCapWithinKernelBound` asserts two constants and is documented as what keeps the ceiling "a FACT" | n/a | n/a | **Fixed 2026-10-01:** `TestAttentionKernelsPastTileBound` runs the three decode attention kernels past 4096 keys against a float64 reference; the prefill half is `TestPrefill_exactAttentionDeclinesPast4096Keys` |

---

## 3. Cross-area notes

1. **E-X01 is resolved; the default Metal path is intact.** E found `mc3-prefill-attr-2026-09-28.md:27` saying "since
   9ccf7fb1 the int4-embedding default makes the Metal resident decline to the CPU" and could not read `loadflags`. In
   the working repo `internal/loadflags/loadflags.go:62-74` registers `-backend` with default `cpu` (`auto` since R17, 2026-10-01) and sets
   `EmbedInt4 = true`, and `:122-131` (`embedInt4()`) returns false for `Backend == "metal"` unless `--embed-int4` was
   given on the command line; the comment there records the same fault (found 2026-09-28, reproduced 2026-09-30 on the
   0.5B, `docs/quantization.md` "Known issue"). So a plain `-backend metal` load keeps the int8 head and stays resident.
   What is left is B-N01: Metal has no int4 head kernel, so an explicit `--embed-int4` on Metal runs on the CPU. That
   makes the "owner decision, 2026-09-28" to default it on a CPU/CUDA/WebGPU decision that Metal has not joined, and
   joining it is an owner call (§10, O3), not a defect.
2. **A-C01 and F-C01 are the same defect**, found independently. Counted once.
3. **Sub-64-token prompts have two remedies, not one.** A-P02 moves the floor behind new pooled-gate cells at K=16/32/48
   (the f16 batched pass, so *gated*); E-P01 routes the same prompts through MC3's bit-identical 8-row step (so no
   gate). They are alternatives with different coverage: E-P01 needs `batchIneligible() == ""` (two or more KV slots,
   dense W4A8, no QK-norm, no windows, no adapter), which holds for the default Qwen2.5-Coder 1.5B and 7B at
   `--kv-sessions 4`; A-P02 covers a single-slot serve and the families MC3 excludes. A-P01's smaller tile helps A-P02
   and 33–64-token passes either way. Order: E-P01 first (§10).
4. **B-P05 and area E are one change.** The decode down kernel and MC3's `mc3_mma_btd` reproduce the same per-word float
   chain on purpose (MC3 record `concurrency-mc3-s0-2026-09-26.md:80`). Moving decode's down projection to the SA body
   changes bits and breaks decode/verify agreement unless the MC3 down kernel moves with it: do both or neither.
5. **B-P03 moves bits and a golden.** Lowering `attnFADepthFloor` changes the kernel at depths 256–1535, so logits
   there change bitwise; the snapshot golden straddles the floor on purpose (`metal/snapshot_golden_test.go:70-72`, "1534
   declines, 1535 engages") and would be re-baked (G-04's shape), and the R17 set-B gate has to be run at the chosen
   floor. The batched step's per-row attention floor (E-P05) reads the same constant.
6. **`attnFACoreCount = 14` is live more often than the reports say.** C-D01(3) and E-D01 call it stale against the
   16-core target and, in E's words, dead on the shipped path. That holds for G=6,7 (the fixed split of 16 overrides it,
   `metal/model.go:2889-2892`). `attnFASplitFor` still sizes the split from it (`want = ceil(2*14/nKV)`) for every other
   group size, which is exactly the population B-P02 is about; settle the constant in B-P02's probe.
7. **C-C01 and F-C02 compound.** The auto-pin that C-C01 describes (a guard that trims on a tight machine hands Metal a
   context of 12–30k positions) is what makes F-C02's `sc[4096]` overrun reachable on an unpinned load. Fix the planner
   and the kernel guard together.
8. **The shared-event design (M-11, G-05, C-B03) has two reviewers agreeing its premise is stale.** C: the primitive
   tested was the event, not MLX's spin fence, and the target cost changed by an order of magnitude after the slot
   residency set. D: 61 × 14 ms = 854 ms cannot be right against a recorded 128–167 ms M26 paged token, so the average
   buffer costs at most 2.1–2.7 ms. The cheapest next step is D-P02's `GOINFER_MOE_PROF_SPLIT=1` split, before any design.
9. **Concurrent dispatch (C) and E-P05 do not disagree.** C finds no independent siblings on the dense chain (0 to
   +0.3%); E finds them in the batched step, where each row's attention is its own dispatch pair, and notes aikit does
   not expose `computeCommandEncoderWithDispatchType:`. The multi-row kernel reaches the same occupancy without it.
10. **Stale text clusters in `docs/benchmarks.md`.** A-D01, B-D01, F-D01 and D-D01 all point at lines `:43`, `:47`,
    `:1026-1031`, `:1080-1081`, `:1976-1985` and `:2260-2262`: one pass over that file closes the stale-claim findings A-D01, B-D01, F-D01 and D-D01 items 4 and 5 (§10, T0.5).
11. **What was re-read at consolidation.** *A-C01/F-C01:* `prefillOK` (`metal/model.go:1027-1029`) has no `kvI8` term, `prefill.go` and
    `PrefillLast` never mention it, `kv_store_f16` writes `device half*` at `pos*kvDim + i` (`metal/prefill.go:676-683`), and the
    allocation under `r.kvI8` is `paddedCtxCap*kvDim*1` bytes plus separate scale buffers (`metal/model.go:1308-1326`), so
    positions at or past ctxCap/2 write beyond the buffer and every earlier position is the wrong layout. *F-C02:*
    `threadgroup float sc[4096]` at `metal/prefill.go:363`, `sc[s]` written with the absolute key index at `:286` and `:298`; the
    dispatch falls to that kernel when `!useFusedAttn` (hd above 128, hd not a multiple of 8, or the env knob off,
    `:1041-1085`), and `ctxCap` reaches 32768 for an explicit `-ctx` or a guard-pinned load (`metal/model.go:22-62`). *B-P01:*
    `canUseAttnFA` returns false unless `g.hd == 128` (`metal/model.go:2882`). *C-P01:* `gScaleCache`/`dScaleCache` are built
    from `int4DirectBytes` and kept for the model's life (`metal/gemma4_moe.go`'s paged branch at `844700f8`; removed 2026-10-02). *E-P01:* `PrefillLast` declines when
    `startPos+len < floor` (`metal/backend.go:782-797`) and `PrefillLastNArgmax` already steps same-slot positions in 8-row
    pieces (`metal/batch.go:772-803`). None of this was run on a device.

---

# Part II — Area reports

The six reports as the reviewers wrote them, edited only for scratch path names, cross-references and the E-X01 resolution (§3 note 1). Within an area, `(a)` is the summary, `(b)` the findings table in order of effect, `(c)` the full entries, `(d)` the carry-forward of that area's Sep 12 IDs (merged into §12), `(e)` what was checked and found correct, `(f)` what could not be settled statically (consolidated in §13). MLX citations are relative to `mlx/backend/metal/` at `9c3d355`.


## 4. Area A — Prefill: GEMM, prefill attention, the sequential/batched split, chunked prefill, floors

Snapshot: goinfer 844700f8, MLX 9c3d355. Static review, nothing built or timed.
Number provenance tags: [rec] a figure the repo recorded (doc:line), [cnt] counted from code shape, [mlx] MLX source
(file:line), [proj] projection with band and basis.

### (a) Summary

1. A-C01 (Critical, only with `-kv i8`): Metal `PrefillLast` has no int8-KV guard and writes f16 K/V into the int8-sized cache; one line in `prefillOK`.
2. R16 and R19 sit at or past the MLX shapes (64x64 = MLX medium steel tile, above `qmm`'s 32x32; R19 = steel bq32/bk16/wm4/wn1). No swizzle exists on M1 Pro, and MLX steel does not share K/V across GQA heads. Little is left to borrow in prefill: shape-keyed tiles, split-K (fidelity-gated), a fragment-direct epilogue.
3. A-P01 (Major): the record's pass-cost table fits `T = 18 ms + 67.9 ms per 64-row tile-row`; the first tile-row carries ~18 ms of idle-core penalty and a <=32-row turn pays 64 rows of MMA. BM=32/BN=32 tiles chosen by shape are bit-identical.
4. A-P02 (Major): the floor of 64 is a gate floor, not a speed floor; served K=32 batched was 2.51x faster and never gated; fresh prompts of 10-63 tokens still run at ~10.5 ms/token.
5. A-P03: the depth rows of the pass-cost table predate R19's wiring and measured the fused kernel; R19's grid is 12-24 threadgroups at M<=64 on 16 cores; one zero-code rerun settles it.
6. A-P04: `HiddenLast` (/v1/embeddings) is still one synchronous command buffer per token.
7. Stale: benchmarks.md:43 K=3900 row and "attention untouched" clause (R19), a cited record missing (r3-startpos), four comments in backend.go, and G-08's closure test runs the fused kernel, not steel.

### (b) Findings table (ordered by effect)

| ID | Sev | Claim | Evidence | MLX analog | Band | Probe |
|---|---|---|---|---|---|---|
| A-C01 | Critical (non-default `-kv i8`) | Batched prefill writes f16 K/V into int8-allocated KV and attention reads it as f16 | metal/prefill.go:1000,1196; metal/model.go:929,1302-1306; metal/backend.go:744-816; decoder/model.go:1583-1590 | none | n/a | decline test + one-line guard |
| A-P01 | Major | <=64-row passes run o/down/qkv at 24-32 threadgroups and a 64-row MMA tile for <=32 real rows | metal/prefill.go:1227-1230; chunked-prefill-2026-09-27.md:102-105; concurrency-mc3-s0-2026-09-26.md:34-39 | quantized.cpp:1087-1090,1152-1176; matmul.cpp:166-170 | C<=32: -15..-40 ms of 96-105; C 33-64: -5..-18 ms [proj] | small-M GEMM arms + TestMC5_passCost |
| A-P02 | Major | Floor 64 is a gate floor; fresh 10-63-token prompts stay sequential; K=32 batched was 2.51x faster, ungated | metal/backend.go:626,772-775; metal-prefill-floor-2026-09-20.md:49-55,73-75 | quantized.cpp:89-130 | K=32 TTFT ~305 -> ~120 ms [rec]; K=16 ~1.6x [proj] | pooled-gate cells K=16/32/48 |
| A-P03 | Major if confirmed | Small-M prefill attention at depth is probably occupancy/latency-bound; the table that says so predates R19 | chunked-prefill-2026-09-27.md:102-105; metal/prefill.go:1289; metal-prefill-attn-2026-09-27.md:91-117 | scaled_dot_product_attention.cpp:642,951,1001 | 0 or -15..-30 ms/pass at depth 2048 [proj] | rerun TestMC5_passCost post-R19 |
| A-P04 | Major (owner call on exactness) | `HiddenLast` is K synchronous command buffers | metal/backend.go:891-931; metal/model.go:1821-1840 | none | pipelined 1.05-1.5x; batched ~10x at K=512 | K=256 embeddings wall, byte compare |
| A-B01 | Minor | MLX split-K (`qmm_splitk`, ~512 TGs) would fix o/down occupancy but reorders the reduction and, keyed on M, breaks chunk invariance | quantized.cpp:1152-1176,1890-1900 | same | not recommended before A-P01 | only if BN=32 under-delivers |
| A-B02 | Minor | Epilogue goes through threadgroup scratch with 2 barriers per accumulator; MLX stores from fragments | metal/prefill.go:62-76,136-139; metal-prefill-gemm-s2-2026-09-25.md:69-70 | mma.h:540,577 | 2-5% of GEMM [proj] | A/B epilogue on `s2` harness |
| A-B03 | Minor (unmeasured) | hd=256 (Gemma 3 <=12B) prefill attention stays on the scalar exact kernel | metal/prefill.go:1073-1074; audit M-06 closure | scaled_dot_product_attention.cpp:515-516 (bd256, wn=2) | unknown | decomp run on gemma3-1b at K=2048 |
| A-P05 | Minor | Decoder switches to PrefillLast at suffix>=8; whole-pass break-even is ~9-10 tokens today | decoder/model.go:1583; counted below | quantized.cpp:89-130 (6/10/14 on M1 Pro) | <=20 ms on 8-9 token turns | revisit after A-P01 |
| A-D01 | Minor | benchmarks.md:43 K=3900 row and its "attention untouched" clause are stale after R19 | benchmarks.md:43; red-october.md:433 | | | one served TTFT cell |
| A-D02 | Minor | Stale comments: "256 under M-02", "decoder always passes 0", "needs the chunking cuda has", test comment "(512)" | metal/backend.go:764-768,772-774,778-781,814-815; metal/prefill_ttft_test.go:42 | | | |
| A-D03 | Minor | r3-startpos-speed-2026-09-21.md is cited and absent; its 4.01x/4.13x predate R16 | red-october.md:859-868 | | | |
| A-G01 | Minor | `attention_prefill_steel` has no direct test; floor logic has no test; G-08's closure covers the fused kernel only | tests listed in entry | | | |
| A-C02 | Minor | No finite check on PrefillLast logits (f16 residual) | metal/backend.go:763-862; decoder/model.go:1583-1595 | | guard ~0.05 ms | |
| A-N01 | Minor | Per-call scratch and lazy compile (N-15, N-16) stay open; fixed per-pass cost is ~18 ms | metal/prefill.go:962,1060,1098-1126 | | <=3 ms/pass [proj] | |

### (c) Full entries

#### A-C01 · Critical (only with `-kv i8`, a non-default flag): batched prefill is not guarded against the int8 KV layout (the same defect as F-C01, found independently)

- Where: `metal/model.go:929` (`r.kvI8 = m.KVCacheI8()`), `:1144-1148` (with int8 KV, `kvBytes = paddedCtxCap*kvDim*1` and the
  per-position scale buffers `ks/vs`), `:1162-1163` (`kc/vc = byteBuf(kvBytes*allocSlots)`). `metal/prefill.go:1000` binds
  `pKv = kv_store_f16`, `:1077` dispatches it on `r.kc[l]`, and `:1078-1084` run attention kernels whose KV arguments are
  `device const half*`. `prefillOK` (`metal/model.go:1027`) lists features, per-layer geometry, Gemma-4 MoE, paged MoE and DeltaNet;
  `metal/backend.go:744-816` adds enable, floor and cap checks. Nothing mentions `kvI8` (grep of `kvI8` in `metal/prefill.go` and the
  `PrefillLast` guards returns only `UploadKV` at `metal/backend.go:1074,1098`). `decoder/model.go:1583-1590` calls `PrefillLast` for any
  suffix >= 8 and `decoder/residency.go:1272-1273` and `metal/residentkv_alloc_test.go:77` show Metal reports and runs `"i8"`.
- Failure (derived, not run): any prompt that reaches `PrefillLast` (>= 64 tokens, no adapter) on a Metal resident loaded with
  `-kv i8`. The scatter writes 2 bytes/element into a 1 byte/element buffer: positions >= ctxCap/2 write past the allocation
  (1.5B, kvDim 256, ctx 4096: a 3900-token prompt writes ~2.0 MB into a ~1.05 MB buffer; UMA does not fault, it corrupts a
  neighbour); below that, the layout is wrong and `ks/vs` are never written, so every later `attention_i8` step reads zero scales.
  Output is wrong, silently.
- Why tests do not see it: `TestMetalBuildResident_KVI8` (`metal/kv_i8_test.go:269-330`) runs `Forward` for 5 tokens at ctx 64; the
  floor would decline a 64-token prompt anyway. `TestUploadKV_matchesSequentialForward_KVI8` covers `UploadKV` only.
- Fix: `&& !r.kvI8` in the `prefillOK` expression (`metal/model.go:1027`), so `PrefillPath()` also reports sequential. Add a test that
  builds an i8 resident and asserts `PrefillLast(embs[>=64], 0)` returns the decline error, run with the floor knob at 0.
- Bit-identity: not applicable (declines to the existing sequential path).
- Why not already closed: Sep 12 audit M-01..M-06/C-01 closures and R3/R4/R16/R19 (red-october.md) never mention int8 KV and
  prefill together; the only int8 prefill text is the CUDA `kvI8` entries in `decoder/prefill_budget.go:120,169` (memory estimate).
- Confidence: confirmed by reading (no guard exists); the failure mode is derived.

#### A-P01 · Major: small-M passes pay a 64-row tile and run the o/down/qkv GEMMs at 24-32 threadgroups

**Status, 2026-10-03 (branch `metal-audit`): shipped.** Graded pass wall shipped ÷ selector at startPos 64, C = 32:
**1.804×** on the 1.5B with both rules (bm32 alone 1.487×, bn32 alone 1.152×), 1.553× on the 7B (where bn32 adds
nothing). `TestGemmTile_bitIdentical` passed first. The GEMM alone: gate|up M = 32 1.61–1.62× on both models.

What goinfer does. `gemm()` (`metal/prefill.go:1227-1230`) dispatches `(ceil(N/64), ceil(Mpad/64))` threadgroups of 128 threads, each a
64-feature x 64-token tile, 4 simdgroups of 32x32, 16 accumulators (`metal/prefill.go:78-140`). Rows past M are staged as zeros and their
MMAs still run (`:36-38`, `:118-125`).

Counted, 1.5B (H 1536, I 8960, qkv 2048; 16-core GPU), any M <= 64:

| GEMM | N | K | threadgroups | K slabs per threadgroup |
|---|---|---|---|---|
| qkv | 2048 | 1536 | 32 | 48 |
| o | 1536 | 1536 | 24 | 48 |
| gate/up | 17920 | 1536 | 280 | 48 |
| down | 1536 | 8960 | 24 | 280 |

Recorded: S0 measured the GEMM flat across M = 1-64, 1.5B gate/up 1.29-1.34 ms at every M, 12-14x one decode GEMV
(concurrency-mc3-s0-2026-09-26.md:34-39). The pass-cost table (chunked-prefill-2026-09-27.md:102-105), startPos 64: C=16 95.8 ms,
C=64 104.8, C=128 154.2, C=256 290.0.

Counted from that table: 128->256 adds 135.8 ms for two tile-rows, 67.9 ms per tile-row, which matches the FLOP count at the
recorded 2.7-2.9 TFLOPS (1.31 G weights x 2 x 64 rows = 167 GFLOP = ~62 ms; the record itself says "~70 ms per 64 tokens",
chunked-prefill:109). Fitting `T = F + 67.9 * tile-rows` through C=256 gives F = 290.0 - 4 x 67.9 = 18.4 ms; C=128 then sits on
the line (18.4 + 135.8 = 154.2). The first tile-row costs 104.8 - 18.4 = 86.4 ms against 67.9, so ~18.5 ms of the first tile-row is
not FLOPs: it is the 24-32-threadgroup GEMMs (qkv, o, down = 41% of the weights, counted 3.1+2.4+13.8 of 46.7 M per layer) leaving
cores idle, and, for C <= 32, 64 rows of MMA work for at most 32 real rows. A second tile-row fills those cores and costs less
(49.4 ms for C=64 -> 128 against 67.9 saturated). F = 18.4 ms is the fixed cost of everything else (attention at depth 64,
norms, rope, kv store, LM head, encode, commit, scratch), which agrees with the decomp record's 3-10 ms for the small kernels at
K=512 (metal-prefill-decomp-2026-09-25.md:64-66).

Premise conflict. `mc3-prefill-attr-2026-09-28.md:70` reads the 125 ms/pass average as "per-pass overhead, not the prefill compute
itself". The fit above says the fixed overhead is ~18 ms; the rest of a <=64-token pass is GEMM at one tile-row.

MLX analog. `QuantizedMatmul` picks `qmm` at `M >= get_qmv_batch_limit` (`quantized.cpp:89-130`, M1 Pro default branch: 14 for
D,O <= 2048, 10 for <= 4096, else 6; dispatch at `:1890-1900`) and runs a fixed 32x32 tile, wm=wn=2, 16x16 per simdgroup
(`:1087-1090`). Small-M waste is bounded by the small tile, and occupancy comes from `qmm_splitk`, which targets ~512 threadgroups
(`:1167-1176`). The `bm = M <= 32 ? 32 : 64` rule exists only in the NAX path (`:840`; M5, out of reach). The dense steel table
(`matmul.cpp:90-171`) gives M1 Pro ('s', the "medium device" branch at `:166-170`) 64x64x16, wm=wn=2, so R16 equals it.

Proposal (bit-identical): two more pipelines from the same source with compile-time tile constants (function constants or macros),
selected in `gemm()` by shape:
- BM=32 when `Mpad <= 32` (4 simdgroups, each 16 tokens x 32 features or 32 tokens x 16 features; constant trip counts, acc[8]);
- BN=32 when `N <= 2048 && Mpad <= 64` (o and down 48 threadgroups, qkv 64).
`metal/prefill.go:120-127` records that skipping MMAs inside the BM=64 kernel was 5.6x slower (runtime bound, acc[] spilled) and 37%
slower (uniform predicate). Those negatives are about runtime skipping in one kernel; a compile-time smaller tile keeps constant
trip counts, which is what MLX does.

Bit-identity. Each output still accumulates K in the same ordered 8-wide chunks into an f32 `simdgroup_float8x8`
(`metal/prefill.go:41-45`); BM/BN change which threadgroup owns an element, not its reduction order. R16 itself changed the simdgroup tile
and measured bit-identical to the retired kernel (metal-prefill-gemm-s2-2026-09-25.md:136,207). Chunk invariance is therefore
kept, but `TestMC5_prefillChunkInvariance` uses C in {64,128,256,384} (`metal/mc5_chunk_invariance_test.go:65`; 100 and 77 were added on 2026-10-01, F-G02, 81 on 2026-10-02 for E-P01, and 16/32/48 the same day for A-P01); it must add C in {16,32,48} so the
test crosses the selector.

Band [proj]: C <= 32: -15..-40 ms of 96-105 ms (occupancy penalty <= 18.5 ms plus roughly half the MMA work of the 62-70 ms GEMM,
times a 0.75-0.85 per-MMA efficiency factor from R16's prototypes: 16 accumulators/simdgroup 3.25x vs 2.45-2.8x for the others,
gemm-s2 record). C = 33-64: -5..-18 ms (occupancy only; the BN=32 kernel has a lower MMA:load ratio). On the 4-client W7 cell this
is small (24 passes x ~25 ms = ~0.6 s of 19.8 s = ~3%, which is why the owner closed the per-pass cut); on per-turn TTFT of a
25-50 token suffix it is 15-35%.

Probe: extend `TestMC3S0PrefillGEMMSmallM` (`metal/gemm_smallm_mc3_test.go`, already in tree) with arms production / BM32 / BN32 /
both over the 1.5B and 7B shapes, M in {8,16,24,32,48,64}, comparing every output bit; then `TestMC5_passCost` (startPos 64 and
2048, C = 16/32/48/64, median of 5) with the selector wired. Ship: >= 1.15x pass wall at C=32 and zero differing bits in the GEMM
arm, the chunk-invariance test with C=16/32/48, and `TestPrefillRefIdentity`. Kill: gate/up at M=32 under 1.15x production, or any
bit difference.

Why this is not already closed: the pass floor is named "the prefill GEMM's 64-row tile" in chunked-prefill-2026-09-27.md:108-110
but only the tail rule and chunk size were changed; `mc3-prefill-attr-2026-09-28.md:73` closes "the per-pass prefill cost cut" on a
4-client wall-clock share (0.151), not on single-turn TTFT and not a tile change; `concurrency-mc3-s0-2026-09-26.md:34-39` says
the 64-row GEMM "is not a starting point" for batched decode, not for short prefill; `metal/prefill.go:120-127` closes runtime MMA
skipping only. The Sep 12 §7 list does not contain a small-M prefill tile.

#### A-P02 · Major: the floor is a gate floor; fresh prompts of 10-63 tokens still run sequentially

**Status, 2026-10-04: shipped in part.** The floor is 16 (the pooled gate ships on K = 16–64; the pass is 3.48x the sequential loop at K = 16 on the 1.5B). A resident with the batched step keeps the exact step below 64: the graded bound was 32, held back by an open MC3 identity defect (docs/tasks/task-metal-audit-2026-10.md, "A-P02: SHIPPED in part").

Where: `metal/backend.go:626` (`metalFastPrefillFloor = 64`), `:722-724` (floor on `startPos + len(embeddings)`), decoder threshold
`decoder/model.go:1583` (suffix >= 8).

What the floor means. R3 gated K=64 and K=128 only; its registered candidates were {64,128}
(metal-prefill-floor-2026-09-20.md:73-75), and K=32 is "informative only: no fidelity cell was gated there" (:55). The same table:
K=32 batched 263.4 tok/s vs sequential 104.9 (2.51x), K=64 3.83x, K=128 4.66x (:49-51). So 32 tokens cost ~305 ms sequential and
~121 ms batched [rec]. `metal/backend.go:614-619` says going below 64 "needs one more passing decision cell"; no record in
docs/measurements ran one (`GATE_DECISION_KS` appears only in the floor record, and K=32 only as "informative only"), and the harness already takes the variable
(`metal/prefill_gate_ref_test.go:77-92`, default cells 256/512/1024 plus 3900 at `:132-133`).

Costs the sequential path pays that the batched path avoids [cnt]: the weights (~0.9-1.0 GB for the 1.5B at int4) are streamed once
per token rather than once per pass; 10.46 ms GPU per token (R18b figure, concurrency-mc3-s0 §5) so 32 tokens = 335 ms; under
MC3 the whole sequential prefill sits inside `prefillExclusive` (`decoder/model.go:1696`), so other decoders stall 3-4x
longer than a ~100-120 ms batched pass would. Costs it does not pay: no LM head on non-final tokens (M-01 closed, `metal/backend.go:593-609`),
and encode-ahead overlap (`metal/model.go:1955-1970`).

What share of real chat turns lands there. The repo has no prompt-length distribution; I cannot give a share. What is countable:
the Qwen2.5 chat template with its default system prompt is roughly 30 tokens before the user text (hand count of the template,
+-4; verify with the tokenizer), so a first message under ~34 tokens (a one-line question) is below 64 in total. Later turns of the
same conversation have `startPos + M >= 64`, so the floor never applies; their path is the decoder's >= 8 rule. Recorded turn sizes
on the prefix-reuse path are 25 mean (W7: 609 tokens in 24 passes, mc3-prefill-attr-2026-09-28.md:68) and 45-51 (agent loop,
`decoder/resident_reuse.go:19`), so most reuse turns are above 8 and take the batched pass.

Proposal: add K in {16, 32, 48} to the pooled-gate decision cells (needs CPU reference files for those K; the Mac set holds K=64/128/
256/1024/3900, gemm-s2 record:250) and move the floor to the smallest passing K, bounded below by A-P05's break-even.
Band [proj]: K=32 fresh prompt TTFT ~305 -> ~120 ms (recorded ratio 2.51x, measured before R16, so the post-R16 batched side is
no worse); K=16: sequential ~167 ms (16 x 10.46) vs one pass at the 96-105 ms floor, ~1.6x; K=8-10: no gain. After A-P01 the batched
side at K<=32 drops further.
Fidelity: not bit-identical (f16 activations vs the sequential W4A8 path); it needs the pooled §3.2 gate at those K. The alternative
that needs no §3.2 gate, routing sub-floor prompts through MC3's bit-identical 8-row step (S0's `mc3_mma_bt`), is area E's call.
Kill: pooled gate fails at K=32, or served K=32 gain < 1.5x.

Why not already closed: the floor record (:55,:73-75) and the code comment (`metal/backend.go:614-619`) leave it as "one more cell";
`red-october.md:859-868` closed the startPos axis at >= 64-token turns only.

#### A-P03 · Major if confirmed: small-M attention at depth, and a stale table

Evidence. Pass-cost at startPos 2048 (chunked-prefill-2026-09-27.md:102-105): C=16 180.4, 64 265.1, 128 318.4, 256 599.7 ms; minus
the startPos-64 row: +84.6, +160.3, +164.2, +309.7 ms. The table was taken no later than ~05:00 PDT on 2026-09-27 (the revised
candidate it led to was registered at b69c73ad and timed 04:58-05:03 PDT); R19's set-A gate ran 2026-09-27 18:29-19:01 PDT and the wiring
followed (metal-prefill-attn-2026-09-27.md:91-117), so the table measured `attention_prefill_fused`, whose grid is nH x ceil(M/8) simdgroups, 4 per threadgroup (`metal/prefill.go:1234-1237`): 24
simdgroups = 6 threadgroups at M=16.

Current kernel. `attention_prefill_steel` runs `nH * ceil(M/32)` threadgroups (`metal/prefill.go:1289`, `:463`): 12 at M <= 32, 24 at
M <= 64, on 16 cores. Each threadgroup sweeps (startPos+M)/16 key blocks serially, with two barriers per block and a synchronous
device->threadgroup stage of K and V (`metal/prefill.go:595-661`), no prefetch. With one threadgroup per core there is nothing to hide
that latency.

Counted: R19's saturated rate. 618 ms attention at K=3900 over 28 layers = 12 heads x 122 q-blocks x ~122 average key blocks x 28
= 5.0 M threadgroup-blocks, 123 ns per threadgroup-block aggregate. At depth 2048 and C=32 the sweep is 12 x 132 x 28 = 44 k
threadgroup-blocks = 5.5 ms if throughput-bound. If a lone threadgroup per core costs 4-8 us per block (no hiding), the same pass
costs 15-30 ms [proj, the 4-8 us is an assumption]. The pre-R19 +84.6 ms at C=16 is the number to beat.

MLX analog. For qL <= 8 and qL x gqa <= 32 MLX uses `sdpa_vector` (shares one K/V sweep across the packed rows and heads) and a
2-pass key-split form for long keys (`scaled_dot_product_attention.cpp:951,1001`); for qL > 8 it uses the same steel kernel with
grid (NQ, H, B) (`:642`), so MLX has the same occupancy shape at 9-64 rows and offers no ready fix.

Probe (no code): rerun `TestMC5_passCost` (exists, `metal/mc5_chunk_test.go:86-122`, `GOINFER_METAL_MC3=1`) on the current build, adding
startPos 8000 and C = 32/48. Read delta = (startPos 2048) - (startPos 64) at C=32. <= 10 ms: close, the premise is gone. >= 25 ms:
build a BQ=16 steel variant (2 simdgroups per threadgroup, 24 threadgroups at M<=32; bit-identical to the BQ=32 kernel when
`window == 0`, because every row's key-block grid is aligned at 0 and rows are independent; with `window > 0` the block grid
follows the threadgroup's first row (`metal/prefill.go:570-574`), so windowed layers would change bits) or a key-split pass with a
combine (reorders the softmax: pooled §3.2 gate).
This revisits the Sep 12 §7 closure of "split-KV / dedup attention variants": that closure concerns decode attention (one query
row); the premise here is a different kernel at M = 9-64.

Why not already closed: chunked-prefill-2026-09-27.md:111-113 attributes the non-tile cost to "attention run in more, smaller
dispatches" and says "it was not separated further"; R19's gates (metal-prefill-attn:79-95) are at K=512 and K=3900 whole-prompt,
M = K.

#### A-P04 · Major (owner call on embedding exactness): `HiddenLast` is K synchronous command buffers

Where: `metal/backend.go:891-931` (loop calling `forwardHiddenNoHead` per token), `metal/model.go:1821-1840` (`e.End()` per token, so each token
is a commit-and-wait), against `ForwardEmbNoLogitsPipe` (`metal/model.go:1955-1970`, encode-ahead) that `ForwardNoLogits` uses for the
same trunk (red-october.md:868-881 records that the noHead pipeline shipped for KV-only prefill).
Recorded: Sep 12 N-25, PARTIALLY CLOSED 2026-09-13: doc rewritten, batched implementation "left as follow-up"; ~13-18 ms/token,
7-9 s at 512 tokens (pre-R18), batched trunk ~1.8 s then (pre-R16/R19; served K=512 is 613 ms after R16, benchmarks.md:43).
CUDA keeps HiddenLast on exact kernels deliberately (prefill-chunk-demotion-2026-09-21.md:9,21,59) and the CPU path says "never
fast: exact HF parity is the point" (`decoder/embed.go:98`), so exactness for embeddings is a recorded decision, not an oversight.

Two options.
1. Bit-identical: run tokens 0..K-2 through `ForwardEmbNoLogitsPipe` and only the last through `forwardHiddenNoHead`. Same kernels as
   the sequential trunk, pinned byte-identical by `metal/kvonly_prefill_test.go:19-75`. Band [proj]: 1.05-1.5x on K=256 wall; basis: the
   13-18 ms/token record against 10.46 ms/token decode GPU time today; the low end covers R18 having closed part of that gap.
   Probe: K=256 embedding wall before/after, output vector byte compare. Kill: < 1.1x or any bit difference.
2. Batched trunk (`PrefillLast` minus the last two dispatches): ~10x at K=512 [proj from records: ~5-9 s -> ~0.6 s], needs an owner
   decision to give up exact embeddings and a cosine gate against today's `HiddenLast` as oracle.

Why not already closed: N-25 closure text and the `HiddenLast` doc comment (`metal/backend.go:887-895`) both defer it; no later record
touches it (grep of `HiddenLast` in red-october.md and measurements finds only the CUDA and MC1 lines above).

#### A-B01 · Minor: MLX split-K for the o/down shapes

MLX: `qmm_splitk` picks `split_k = max(1, 512 / current_tgs)` with 32x32 tiles, rounds it down to divide K by 32-aligned groups,
and writes f16 partials (x.dtype) into a [split_k, M, N] temp before a reduce (`quantized.cpp:1152-1215`); it runs first when
`M >= vector_limit` and B == 1 (`:1890-1900`). For down at M=32 that is split 10 (K=8960 divides by 320).
goinfer: none. Fidelity: splitting K reorders each element's reduction, so bits change at every M where it engages, and an M-keyed
split makes the result depend on the chunk size, which `TestMC5_prefillChunkInvariance` pins. A split keyed only on (N, K) would keep
invariance but would also run at M=3900, adding partial traffic (3900 x 1536 x 4 B x S) and needing a new pooled gate on every
checkpoint. Recommendation: do A-P01's BN=32 first (bit-identical, 48 threadgroups for o/down); consider split-K only if down is
still the long pole after it. Band: not stated; needs A-P01's data.

#### A-B02 · Minor: store the epilogue from fragments

goinfer stores each 8x8 accumulator to threadgroup scratch, barriers, then 2 scalar loop iterations per lane with the bias or
residual (`metal/prefill.go:62-76`, 16 times per simdgroup at `:134-137`). MLX stores from the fragment lane map (`mma.h:540,577`,
`BlockMMA::store_result`), 2 adjacent elements per lane. goinfer already depends on that lane map (`mc3_frag`, used by R19 and the MC3
kernels). Both the bias and the residual add happen in the same order as today (mode is exclusive per call), so the f32 value and the
f16 conversion are unchanged: bit-identical. The R16 record chose the scratch epilogue deliberately because a peer "writes f32 and
fuses nothing" (gemm-s2 record:69-70); a fragment-direct fused epilogue was not tried and is not a recorded negative.
Band [proj]: the epilogue is 16 stores + 32 simdgroup barriers + scalar half stores per simdgroup against 48 slabs x 64 MMAs for
K=1536 (qkv, o, gate/up) and 280 slabs for down: 2-5% of those GEMMs, < 1% of down, so 1-3% of a pass. Probe: A/B on
`prefill_gemm_s2_test.go` (the retired kernel is kept there for bit compare). Kill: < 1.5% on qkv/o at M=64 or any bit change.

#### A-B03 · Minor, unmeasured: hd=256 prefill attention (Gemma 3 up to 12B)

`useFusedAttn` requires `hd <= 128` and `useSteelAttn` `hd == 128` (`metal/prefill.go:1073-1074`); Gemma 3 1B/4B/12B (hd 256) therefore run
`attention_prefill` (one threadgroup per row and head, scalar). The Sep 12 M-06 closure says the same ("hd=256 fused variant not
attempted"). Recorded ratios at hd=128, K=3900: fused 4.23x over exact (prefill-l2-metal-fused-attn-2026-09-09 title), steel 7.20x
over fused (R19), so roughly 30x between the exact kernel and steel at that head dim; nothing is recorded for Gemma 3 prefill
attention. Five of every six Gemma 3 layers are sliding-window, which bounds their cost; the global layers pay the full K.
MLX instantiates `bd=256` with wm=4, wn=2 (head dim split across two simdgroup columns) and bk=16 on M1 Pro
(`scaled_dot_product_attention.cpp:510-516`; `steel_attention.metal` instantiations). goinfer's R19 holds 16 O fragments and 16 Q
fragments per simdgroup; hd=256 doubles both, which is why MLX splits wn.
Probe: the decomp harness on gemma3-1b at K=2048: attention share of the full replay. If >= 20%, build the hd=256 variant. If
< 5%, close.

#### A-P05 · Minor: where the decoder's >= 8 threshold sits

`decoder/model.go:1583` sends suffix >= 8 to `PrefillLast`. Whole-pass break-even [cnt]: pass floor 96-105 ms (C <= 64, startPos 64)
over 10.46 ms/token = 9.2-10.0 tokens. At M=8-9 on a deep prefix the batched pass is ~12-20 ms slower than sequential, and at
startPos 2048 the pre-R19 numbers (C=16: 180 ms vs 16 tokens x ~11-12 ms) show no advantage until about C=16. MLX's equivalent
crossover on M1 Pro is shape-keyed, 6/10/14 (`quantized.cpp:89-130`); S0 measured the GEMM-only crossover at M ~ 13-16
(concurrency-mc3-s0:36). The threshold is right only by coincidence after A-P01 lowers the floor (~75-85 ms -> ~7-8 tokens).
Re-evaluate once A-P01 lands; no change before.

#### A-D01 · Minor: stale Metal prefill row

`benchmarks.md:43` quotes K=3900 as 1.96x behind Ollama (8.28 vs 4.23 s) "and the K=3900 remainder is attention, which this change
does not touch". R19 then moved in-process `PrefillLast` K=3900 from 8171 to 4316 ms (red-october.md:433; metal-prefill-attn:115-118)
and lists "served TTFT vs Ollama" as owed. If the in-process delta (3.85 s) carries to the served number, ~4.4 s vs 4.23 s, about
1.05x [proj, not measured]. Label the row "pre-R19" until a served cell exists. The short-K rows (K <= 256, TTFT 0.38x-2.3x Ollama in the
Sep 18-20 tables) predate R16 as well and are not re-measured.

#### A-D02 · Minor: stale comments
- `metal/backend.go:782-798`: "A Metal prefill that wants mid-pass cancellation needs the chunking cuda has"; chunked prefill shipped
  (`decoder/model.go:1657-1708`, default 512) but only while another generation is decoding, so a lone long prompt is still one
  uncancellable command buffer (the comment's conclusion holds, its reason is stale).
- `:714` "256 under M-02 before that" and `:719-721` "K=256 itself passed": the floor is 64 and the gated cells are 64/128.
- `:739` "Unreachable today (the decoder always passes 0)": `residentPrefillSeed` passes `from` (`decoder/model.go:1583`), the
  agent-loop reuse shape.
- `metal/prefill_ttft_test.go:42` "(512)"; `metal/prefill_gate_test.go:71-75` "(K=256 ... 256 under M-02)".

#### A-D03 · Minor: missing record
`red-october.md:859` and `audit-metal-2026-09-12.md:1735` cite `docs/measurements/r3-startpos-speed-2026-09-21.md`; the file is not in the
snapshot (the test `metal/r3_startpos_speed_test.go` is). The quoted result (batched 4.01x at a 64-token turn, 4.13x at 128, startPos
512) was measured before R16 and does not cover the 8-63-token range. Restore or regenerate it.

#### A-G01 · Minor: gates that cannot see the default kernel
- `attention_prefill_steel` has no unit test. `attention_prefill_fused_test.go` covers GQA, ragged M, window and startPos but at
  hd 16-128 against the fused kernel; the steel window path (`jStart`, `myWin`, `sStart`, `metal/prefill.go:568-579`), ragged tail
  (`active`, `myRow >= M`) and startPos > 0 have only: `TestPrefillParity` (startPos 0, K=140, cosine >= 0.95 and argmax, heavy and
  Qwen2.5-1.5B only), `TestPrefillNoNaN` (finite), the decomp harness comparing it with the fused kernel (`prefill_attn_r19_test.go`,
  `goinfer_testhooks`), and the opt-in MC5 chunk test.
- G-08's closure test (`metal/prefill_startpos_test.go:32-98`) uses `genTinyWeights` with hd = 16 (`metal/moe_model_test.go:23-26`), so it runs
  the fused kernel, not steel. The only default-kernel evidence at startPos > 0 is `TestMC5_prefillChunkInvariance` (`GOINFER_METAL_MC3=1`,
  `goinfer_testhooks`, one non-windowed checkpoint), which compares chunked against whole bit for bit and carries a control arm
  (`metal/mc5_chunk_invariance_test.go:109-137`); it proves consistency with the startPos-0 result, which the pooled gate does grade
  (`metal/prefill_gate_ref_test.go:561`, startPos 0 only).
- The floor decision has no test: grep of `metalFastPrefillFloorFor` and "prompt too short" in `*_test.go` finds comments only; many
  tests set `GOINFER_METAL_FAST_PREFILL_FLOOR=0` to avoid it. Nothing asserts that `startPos + M < 64` declines, that
  `startPos=1000, M=8` admits, or that the knob parses.
- Fix: a synthetic hd=128 case in `attention_prefill_fused_test.go`'s table run against `attention_prefill` (exact) with window, ragged
  M and startPos; a pure-logic test for the floor predicate.
- Status, 2026-10-01: done. The kernel half is F-G01's `TestAttentionPrefillSteelMatchesFloat64`, which runs steel with
  window, ragged M and startPos against float64 rather than against the exact kernel. `TestPrefillFloor` (default-run)
  checks the parser and the decision through `PrefillLast` on the tiny resident: 63 positions decline and 64 admit,
  cached positions count (56 + 7 declines, 56 + 8 and 1000 + 8 admit), the knob raises the floor or turns it off, and
  `PrefillPath` names it. Dropping `startPos` from the predicate, `<` for `<=`, and accepting negative knob values each
  fail it.

#### A-C02 · Minor: no runtime finite check on batched-prefill logits
The residual stream is f16 (`xF` etc., `metal/prefill.go:1117-1122`); a family whose residual exceeds 65504 gives Inf and NaN logits.
`TestPrefillNoNaN` covers one checkpoint (`metal/prefill_nan_test.go:19-53`); neither `metalResident.PrefillLast` (`metal/backend.go:763-862`)
nor `residentPrefillSeed` (`decoder/model.go:1583-1595`) inspects the 152k logits. A host scan costs ~0.05 ms [cnt: 152k floats];
on NaN, return a decline error so the sequential path re-runs the prompt. Which families overflow is unmeasured here.

#### A-N01 · Minor: per-call scratch and lazy compile
N-15 (26 per-call buffers from `make`d slices) is unchanged at HEAD (`metal/prefill.go:1117-1145`). Sizes [cnt]: ~2.5 MB at M=32 (guF
32x2x8960x2 B = 1.1 MB, dqF 0.57 MB, qkv 0.13 MB, x/norm 0.2 MB), ~15 MB at a 512-token chunk (chunked-prefill:111-113 [rec]). The fit in
A-P01 puts all fixed per-pass cost at F ~ 18 ms, so removing the allocation is worth <= 3 ms of a short pass [proj]; the owner closed the
per-pass cut (mc3-prefill-attr:73) and N-15's own zero-pad-row invariant caveat (audit §6, N-15 entry) still needs a per-kernel
trace. N-16: `ensurePrefill` (`metal/prefill.go:962`) is called only from `PrefillLast` (`:906`), so the first >= 64-token request after
start pays the library compile (now more kernels, with steel); it is unmeasured.

### (d) Carry-forward (Sep 12 audit, area A)

| ID | Status | Evidence |
|---|---|---|
| M-01 | CLOSED-VERIFIED | `ForwardNoLogits` `metal/backend.go:593-609`, pipelined `metal/model.go:1955-1970`, used at `decoder/model.go:1616`; byte-identical tests `metal/kvonly_prefill_test.go:19-75` (tiny fixture). Residual: `HiddenLast` stays synchronous (A-P04) |
| M-02 | CLOSED-VERIFIED, superseded | floor 512 -> 256 -> 64, `metal/backend.go:611-626`; the remaining question is A-P02 |
| M-03 | CLOSED-VERIFIED, superseded by R16 | `metal/prefill.go:78-140`; R16 3.22x recorded (gemm-s2) |
| M-04 | PARTIAL | O-in-registers and wider key tile superseded by R19 (`metal/prefill.go:533-672`). The Fix's third item, packing a GQA group's 6 heads into one threadgroup so K/V load once per group, is not done; MLX does not do it either (`steel_attention.h:97` re-stages per head), R19 record names it "an option neither peer takes" |
| M-05 | PARTIAL | non-paged MoE has an expert-major batched path (`metal/prefill.go:1159-1352`, knob `GOINFER_MOE_EXPERT_MAJOR`); paged MoE and DeltaNet still decline (`metal/model.go:1027`) and run sequentially. Area D owns detail |
| M-06 | PARTIAL | `FeatPerLayerRoPE` declared (`metal/model.go:114`); Gemma 3 reaches the batched GEMM; hd=256 attention stays on the exact kernel (`metal/prefill.go:1073`), now A-B03 |
| C-01 | CLOSED-VERIFIED | `metal/model.go:1309-1316` pads each layer's per-slot allocation to 8 rows, MC3 slots keep their own padded region (`:1152-1163`); steel guards `j < jEnd` (`metal/prefill.go:602`) so only the fused kernel over-reads, into padding |
| C-02 (HiddenLast part) | CLOSED-VERIFIED | paged MoE declines at `metal/backend.go:911` |
| G-01 | CLOSED-VERIFIED | `metal/moe_model_test.go:281-330` rewritten for the admit side, floor knob at 0 |
| G-03 | CLOSED-VERIFIED | `prefill_gemma_test.go` asserts `r.sandwich` and `r.prefillOK` before comparing; bar is cosine >= 0.95 on a 12-token tiny fixture (a smoke bar) |
| G-07 | CLOSED-VERIFIED | `metal/prefill_gate_test.go:63-75` sets the floor to 0; superseded-form note at `:44-49`; comment text stale (A-D02) |
| G-08 | PARTIAL | the closure test runs the fused kernel on an hd=16 fixture; the default steel kernel's startPos > 0 is pinned only by the opt-in MC5 test (A-G01) |
| N-15 | OPEN, deferred | A-N01 |
| N-16 | OPEN | A-N01 |
| N-17 | CLOSED-VERIFIED | only `gemm_w4f16_store` remains in `prefill.go`; the retired R16 predecessor lives in `metal/prefill_gemm_s2_test.go:396` |
| N-18 | CLOSED (by R19) | steel keeps O in registers, no `pTile` reload |
| N-19 | CLOSED by measurement | `rope_f16` still computes `cos/sin` per (row, head, pair) (`metal/prefill.go:317`), but rope q+k, both norms and kv store total 3.19 ms at K=512 (0.2%) and 18.7 ms at K=3900 with the LM head (0.1%) (metal-prefill-decomp-2026-09-25.md:66,81) |
| N-25 | PARTIAL | A-P04 |

### (e) Checked and found correct, and the MLX answers

MLX answers to the brief's questions.
- Tile per shape and generation. Dense steel GEMM: `GEMM_TPARAM_MACRO` (`matmul.cpp:90-171`) keys on device class and dtype: 'g'/'p'
  small (64x32 or 64x64, bk 8/16/32), 'd' ultra (64x64 or 32x64 by problem size, `:111-165`), everything else including M1 Pro 's'
  64x64x16, wm=wn=2 (`:166-170`). Quantized: non-NAX `qmm` is a fixed 32x32x32, wm=wn=2 (`quantized.cpp:1087-1090`), `qmm_splitk` 32x32 (`:1167`),
  gather/MoE `bm = (M/E < 64) ? 32 : 64` (`:1568`), tiny `bm=16,bn=32` (`:1734-1735`), NAX `bm = M<=32 ? 32 : 64` (`:840`, M5 only). The
  vector/matrix crossover is shape- and generation-keyed (`:89-130`).
- Swizzle. `swizzle_log` is 0 on the non-NAX dense path (`matmul.cpp:456`; the heuristic is commented out) and absent from `qmm`; the live
  heuristics (`:295-299`, `:799-801`) are NAX or other kernels. Nothing to borrow on M1 Pro.
- M remainder. `qmm_t_impl` takes `num_els = min(BM, M - y_row)` and calls `load_safe` on the boundary threadgroup only
  (`quantized.h:1257-1277`), still running all BM MMAs. The dense steel GEMM uses function constants `align_M/N/K` (`matmul.cpp:255-262`).
  goinfer uses a per-thread `aok`/`wok` predicate and zero staging (`metal/prefill.go:97-101`); same MMA cost, and N is always a multiple of 64 for
  the shipped shapes (1536, 2048, 8960, 17920, 3584, 4608, 18944, 37888), so `wok` never fails there (a compile-time `align_N` would
  remove a select per slab, a fraction of a percent).
- GQA in the steel attention. One threadgroup per (q block, q head) with `kv_head = head / gqa_factor` (`steel_attention.h:97`,
  grid `(NQ, H, B)` at `scaled_dot_product_attention.cpp:642`); K and V are re-staged for every q head, as in R19. K and V alias one
  threadgroup buffer (`steel_attention.h:134-139`); R19 keeps two buffers (one barrier pair per block against MLX's two stagings).
- What R16 kept and left. Kept: threadgroup-staged operands, ordered K accumulation, in-kernel dequant; the tile equals MLX's medium
  tile and is larger than MLX `qmm`'s (16 accumulators per simdgroup against 4). Left: shape-keyed tile selection and split-K.
- What R19 kept and left. Kept: bq32/bk16/wm4/wn1 (`scaled_dot_product_attention.cpp:510-516`), staged K/V once per block, register O,
  `exp2` with log2e folded into the score scale, causal block bounds. Added: sliding window and startPos masks. Left: function-constant
  specialisation (`align_Q/align_K/has_mask/do_causal`, `steel_attention.h:11-15`) in favour of runtime uniform branches, hd != 128 (MLX
  instantiates 64/72/80/96/128/192 with wn=1 and 256 with wn=2), sinks, array masks, lse.

Verified correct, no finding.
- Floor semantics: `promptLen = startPos + len(embs)` (`metal/backend.go:782`); a reuse turn above 64 total takes the batched pass at any suffix >= 8.
- Chunk invariance and its control arm (`metal/mc5_chunk_invariance_test.go:109-137`): the perturbed-token control sees differences only from position 700.
- Chunk tail rule (`decoder/model.go:1666-1670`): the final pass has 8..C+7 tokens, so `residentPrefillSeed`'s 8-token rule is never hit by a cut.
- A cancelled prefill returns instead of falling to the sequential loop (`decoder/model.go:1585-1592`).
- `PrefillLast` recovers its per-call allocation panics into an error (`metal/backend.go:836-845`), and `startPos < 0` is rejected.
- Steel's tail handling: a simdgroup wholly past M loads no Q, meets every barrier (`metal/prefill.go:581-587`, `:527`).
- MLX's NAX and `*_nax*` paths are out of reach on M1 Pro and are not findings.

### (f) Questions I could not settle statically

1. Post-R19 pass cost by depth at C = 16-64 (A-P03). Zero-code probe above.
2. GPU versus wall split of the ~100 ms pass floor. The `TestMC5_passCost` log prints "(GPU x ms)" but the logs are not in the snapshot;
   the F = 18 ms fit assumes linearity in tile-rows.
3. Whether the BN=32 kernel's lower MMA:load ratio costs more than the occupancy it buys (needs the device).
4. The prompt-length distribution of real first turns, and so the share under the floor (A-P02). Not recorded anywhere I could find.
5. Whether chunk invariance holds for a windowed hd=128 model: the steel key-block grid starts at the threadgroup's first row's window
   start (`metal/prefill.go:570-574`), which depends on chunk alignment; MC5 ran only on a non-windowed checkpoint. No shipped hd=128
   windowed family was identified.
6. Whether any checkpoint outside Qwen2.5-1.5B/7B overflows the f16 residual (A-C02). The pooled gate has graded only those two,
   at startPos 0, K in {256,512,1024,3900}; every other prefillOK family is default-on on the strength of the f16-numerics argument.
7. A-C01's on-device behaviour: the failure is derived from the allocation sizes and kernel signatures.


## 5. Area B — Decode kernels: int4/int8 GEMVs, decode attention, small fused ops

Reviewer B. Snapshot at HEAD 844700f8 (2026-09-30). Static only: no GPU run, no timing, no edits. Every
number is a recorded figure (doc:line), a count from code shape (the count is shown), MLX source (file:line), or a
projection (labelled, with a band and its basis). Target: M1 Pro, 16 GB, 16 cores, ~200 GB/s (S0 streaming ceiling
178-182 GB/s, `metal-decode-gemv-s0-2026-09-26.md:42`).

Tags: P performance, B borrowable MLX technique, D stale doc, N minor. "Record" means a doc under `docs/`. Paths are
relative to the goinfer repository root unless they start with `mlx/` (MLX, under `mlx/backend/metal/`).

---

### (a) Summary

1. Largest open term: decode attention coverage. `attention_fa_blk` (R17) serves hd=128, GQA group 6 or 7, and 1536+ keys only.
   The 0.5B (hd=64, 61% of its 3900-key token), every other group size, and every depth below 1536 run the older kernels
   (B-P01, B-P02, B-P03). The 1536 floor was measured on the kernel R17 replaced.
2. MLX's quantized GEMV form does not attack the "per-weight issue work" term beyond what R18b took. Counted: about 3.8-3.9
   ALU ops per weight in both `sa_rows_acc` and MLX `qdot`, and goinfer issues 1.7-3.3x fewer load instructions per weight.
3. The remaining gap to MLX at depth 128 (0.82x on the 1.5B, 0.75x on the 7B, cross-session; T1.14 same-session: 0.833x and 0.828x) is rate plus about 5.6% of the
   7B token in head bytes (a recorded int8 pin). Candidates inside goinfer: the SA lane mapping's idle tail at K=1536/3584,
   with a bit-identical fix (B-P04); the down projection left on the coal kernel, a REVISIT of M-10 (B-P05); TG geometry (B-P06).
4. Fusion: goinfer already fuses more than MLX (12 dispatches per layer, 11 below the floor). Nothing to borrow.
5. Attention: MLX's `sdpa_vector` forms are the per-key shape R17 replaced, and MLX's partition count (64 or 128) matches
   goinfer's 64 per KV head. Nothing to borrow; R17 is ahead by shape. Two goinfer-internal items: a reduce-scatter for blk's
   QK stage (B-P07) and a depth staircase from the fixed split (B-P08).
6. Carry-forward: M-09, M-10, M-16 NEGATIVE-CLOSED (M-10's premise stale); N-09, N-10 (partial), N-13, N-17, N-30, N-37,
   N-40 checked at HEAD (section d).
7. Every number is recorded, counted, or a labelled projection; no GPU run. `*_nax*` files (M5 only) were not read as findings.

---

### (b) Findings table (ordered by the upper end of the band)

| ID | Sev | Claim | Evidence | MLX analog | Band (projection unless stated) | Probe |
|---|---|---|---|---|---|---|
| B-P01 | Major | hd=64 decode attention has no fa/blk path; on the 0.5B it is 61% of the 3900-key token and still on the per-query-head kernel | `metal/model.go:2881`; `metal/kernels.go:1209,1330`; decomp `:35`; R17 doc `:85`; `docs/benchmarks.md:47` | `sdpa_vector` is templated on D: `mlx/kernels/sdpa_vector.h:15,45`, instantiated 64/96/128/192/256 at `mlx/kernels/scaled_dot_product_attention.metal:59-66` | 0.5B token 1.3-1.85x at 2048-3900 keys; 0 at 128 | hd-templated blk (2 dims per lane), in-sequence attention at 2048/3900 on the 0.5B |
| B-P02 | Major | `attention_fa_blk` is instantiated for G=6,7 only; every other hd=128 GQA shape (for example 32 heads over 8 KV heads, G=4) stays on `attention_fa`, which R17 measured 2.3-3.5x slower | `metal/model.go:1507-1519`; `metal/kernels.go:1430-1433`; R17 doc `:463-466,74-76` | `sdpa_vector_2pass_1` takes G at runtime (`mlx/kernels/sdpa_vector.h:220`, `group_dims(32, gqa_factor, ...)`, `mlx/scaled_dot_product_attention.cpp:817`) | 1.15-1.4x end to end at 2048+ keys on ungraded G (the R17 end-to-end figure at the graded shapes, `benchmarks.md:47`) | instantiate g4/g5/g8, per-head equality test vs g7, in-sequence attention at 2048/3900 |
| B-P03 | Major | REVISIT: `attnFADepthFloor`=1536 was measured on `attention_fa` (R2), not on the block kernel that now runs above it; legacy attention at 1024-1535 keys costs about 3.3-5.0 ms against about 1.8 ms for blk | `metal/model.go:2717-2725,2841,2876`; R17 doc `:465`; `red-october.md:2195-2196`; `metal-depth-r2-2026-09-18.md:55-58`; decomp `:35` | MLX has no floor for its vector kernel; the 2-pass switch is at N>=1024 on arch 's'/'d' (`mlx/scaled_dot_product_attention.cpp:1565`) | 5-20% of the 1.5B token at 1024-1535 keys; 0 elsewhere | floor override hook next to `attnFASplitOverride`; interleaved in-sequence sweep 256..1535 |
| B-P04 | Major | [P,B] SA-family lanes own a whole 32-weight group, so K=1536 takes 2 trips with the second half-idle (1.5 trips of work) and K=3584 takes 4 for 3.5; MLX's 16-weights-per-lane mapping fills all lanes. A bit-identical form exists: exact integer exchange plus an owner-lane float chain | `metal/kernels.go:561,576,578`; R18b doc `:21`; S0 `:48,53`; R18b doc `:61-66` | `mlx/kernels/quantized.h:774` (block_size = 16 x 32 = 512), `mlx/quantized.cpp:488` (fast needs K%512==0) | 1.5B token 3-7% (gate/up 132 -> 150-165 GB/s); 7B 1-3% | synthetic K sweep at fixed bytes, then lane-balanced prototype in `gemv_r18_test.go` |
| B-P05 | Major | REVISIT of M-10: R18b excluded the down projection; it stays on the coal body (per-word float chain, one scale load per word) and is the slowest int4 GEMV, about 84 GB/s (1.5B) and 111 GB/s (7B) against 132 and 166 for gate/up | `metal/kernels.go:622-648`; `metal/model.go:2634-2636,1465`; `red-october.md:2373`; R18 doc `:123-130`; S0 `:49,54`; audit M-10 `:533-564` | `qmv_fast_impl` has one inner loop for every projection (`mlx/kernels/quantized.h:756-821`) | 1.5B 0-9%, 7B 0-10% (needs the fidelity gate; breaks decode/verify order agreement unless E moves the MC3 down kernel too) | existing `gemv_w4a8_sa_resid_rows4` dispatched at the down shape in `TestR18InSequence` |
| B-P06 | Minor | [P,B] every R18/R18b dispatch is 256 threads (8 simdgroups) with a K x 2 B staging prologue and barrier; R18's record names TG size as an untested next suspect; MLX runs 64-thread TGs with no TG memory | `metal/model.go:2609,2630,2960,3069`; R18 doc `:170-175`; `metal/kernels.go:556-557` | `mlx/quantized.cpp:495` (`group_dims(bk, 2, 1)`), `mlx/kernels/quantized.h:769-770` | 0-6% of the 1.5B token (gate/up only); likely <= 3% | TG {64,128,256,512} x R {2,4} at gate/up shapes, then in sequence |
| B-P07 | Minor | blk's QK stage does 32 independent `simd_sum`s per head per block; a reduce-scatter in the measured simd_sum tree order gives bit-identical scores for about 2.6x fewer cross-lane ops | `metal/kernels.go:1357-1362`; MC3 record `concurrency-mc3-s0-2026-09-26.md:74,108`; `metal/batch.go:255-256` | none (MLX does one `simd_sum` per key per head: `mlx/kernels/sdpa_vector.h:278`) | 0-4.6% of the 1.5B token at 3900 keys, 0-3.6% on the 7B | bit-equality test vs shipped blk on 65,536 random vectors, then in-sequence |
| B-P08 | Minor | blk at fixed S=16 costs one trip per 2048 keys: the busiest simdgroup runs ceil(ceil(n/16)/128) trips, so cost steps at 2049, 4097 and so on; the two graded depths (2048, 3900) are the balanced points | `metal/kernels.go:1335,1352`; R17 doc `:76` (1.770 vs 2.537 ms), `:80-83` | `mlx/scaled_dot_product_attention.cpp:776-806` (blocks by arch/N) | step of about 0.77 ms (1.5B, 5.5% of token) or 2.2 ms (7B, 4.8%) at each 2048-key boundary; mean about half | depth sweep, in-sequence, 1536..6144 in 128-key steps |
| B-P09 | Minor | `rmsnorm_quant` makes 3 device passes and 12 TG barriers per dispatch; the last 5 tree levels are inside one simdgroup | `metal/kernels.go:24-64`; decomp `:41` | `mlx/kernels/rms_norm.metal:13-80` (x in registers, one load pass, 3 barriers) | 0.3-2% of the 1.5B token | register-cache variant with the same pinned tree, byte-equal vs shipped |
| B-N01 | Minor | Metal cannot load an `-embed-int4` bundle: `int8Buf` errors on a non-int8 head; there is no int4 head kernel on Metal | `metal/model.go:1358-1365,520-526`; `decoder/weightmat.go:96-106`; `docs/giw-bundles.md:24` | mlx-lm 4-bit checkpoints carry a 4-bit head (not in library source; see f) | 4.4% (7B) to 6% (1.5B) of the token if supported; lossy opt-in, not proposed as a default | none now; see entry |
| B-D01 | Minor | `benchmarks.md:47` says "AHEAD of Ollama in every cell" after R18b; the six graded cells are 1.5B and 7B only, and the 0.5B depth cells (0.75x at 2048, 0.58x at 3900) have not been re-measured since | `docs/benchmarks.md:47`; R18b doc `:104-113` | n/a | n/a | re-run cell g at 0.5B |
| B-D02 | Minor | `model.go` attention-block comment counts 7 dispatches; at fa depths there are 8 (the combine is a separate dispatch) | `metal/model.go:2930-2934`; `metal/model.go:2707-2813` | n/a | n/a | edit comment |

---

### (c) Full entries

#### B-P01 [P, Major] hd=64 never reaches the flash-decode kernels (0.5B and any hd=64 model)

**Status, 2026-10-03: built on main, off by default, pending its night grade.** `attention_fa_blk64` (2 dims per lane,
`half2` loads), instantiated for G = 7. By day it is closer to float64 than the shipped kernel (median relL2 1.66e-7
against 3.80e-7 on the 0.5B), and a one-rep smoke read attention 3.2× / 3.8× faster at 2048 / 3900 keys. The grade,
its rule and the new 0.5B reference cell (Q05) are in the task doc ("B-P01: built, pending its grade").

**Claim.** `canUseAttnFA` declines unless `g.hd == 128` (`metal/model.go:2881`; the comment above it says the
kernel's cooperative-load tiling is fixed to 32 lanes x half4). Both `attention_fa` (`metal/kernels.go:1209`) and
`attention_fa_blk` (`metal/kernels.go:1330`) hard-code `hd = 128u`. A hd=64 model runs the legacy per-query-head
`attention` kernel at every depth.

**Weight of the term (record).** Qwen2.5-0.5B, 1 attention number per depth, `metal-decode-decomp-2026-09-25.md:35`:
0.48 ms of 5.22 at 128 keys (9%), 4.09 of 8.87 at 2048 (46%), 7.61 of 12.40 at 3900 (61%). The same record gives about
6 GB/s KV read on that kernel (`:54`). `benchmarks.md:47` still reads 0.75x (2048) and 0.58x (3900) of Ollama for the
0.5B, from the 2026-09-25 pre-registered cell g; the R17/R18/R18b updates cover the 1.5B and 7B only.

**Why this is not already closed.** Checked `red-october.md:2203-2204` (R17: "the 0.5B (head dim 64): the kernel's
reach there is a scope choice, recorded either way"), `metal-decode-attn-r17-2026-09-25.md:85` ("out of reach there"),
and the model.go comment at `:2551-2556`. It is a recorded scope boundary, not a recorded negative, and no later record
reopens it (grep of `docs/` and `metal/` finds no hd=64 flash-decode work after R17).

**MLX analog.** `sdpa_vector` is a template on D with `qk_per_thread = D / 32` (`mlx/kernels/sdpa_vector.h:15,45,208`),
instantiated for D = 64, 96, 128, 192, 256 (`mlx/kernels/scaled_dot_product_attention.metal:59-66`). At D=64 each lane
holds 2 dims and loads a half2. MLX's own form is the per-key `simd_sum` shape R17 replaced, so the borrow is the D
generality, not the kernel body.

**Delta.** blk with `lane*2` dims (half2 loads, 128 B per key row per simdgroup) keeps the same per-key dependent chain
(G simd_sums) while halving bytes per key, so the speedup over legacy may be smaller than at hd=128. The same template
extension would cover hd=96 (phi3-mini, 3 dims per lane, MHA so G=1) and hd=256 (Gemma, windows, out of scope here).

**Band (projection).** Basis: attention is 7.61 ms of 12.40 at 3900 on the 0.5B. R17 measured blk 2.3-3.5x over
`attention_fa` in sequence (R17 doc `:74-76`) and `attention_fa` 1.26x over legacy at 3900 (metal/model.go:2717-2725 comment),
so 2.9-4.4x over legacy at hd=128. Take 2x as the low case and 4x as the high case on the 0.5B: attention 7.61 ->
3.8 / 1.9 ms, token 12.40 -> 8.6 / 6.7 ms = 1.44-1.85x at 3900. At 2048: 4.09 -> 2.0 / 1.0 ms, token 8.87 -> 6.8 / 5.8 =
1.3-1.5x. Overall band 1.3-1.85x at 2048-3900, none at 128. A projection, not a measurement.

**Bit-identity / fidelity.** Not bit-identical to legacy (block-wise softmax reassociates, as `attention_fa` already
does). Needs R17's precondition 1 gate (teacher-forced, set B, `red-october.md:2191-2194`) run on the 0.5B.

**Probe.** Template `attention_fa_blk` on hd (2 or 4 dims per lane), instantiate G=7 at hd=64, run the
`TestMetalDecodeDecomp` swap (current vs candidate, interleaved, 5 reps x 20 tokens) on the 0.5B at 2048 and 3900 keys
with `attnFASplitOverride` sweeping S in {8, 16, 24, 32} (R17 showed the response is not monotone).
**Kill:** in-sequence attention ratio below 1.5x at both depths (R17's registered kill line, `red-october.md:2178-2185`),
or fidelity gate failure.

---

#### B-P02 [P, Major] The block kernel is instantiated for two group sizes only

**Status, 2026-10-03: built on main, off by default, pending its night grade.** `attention_fa_blk` is instantiated at
G = 2, 3, 4, 5 and 8, and agrees with g7 head for head, bit for bit (`TestAttnFABlk_anyGMatchesG7`), so the new sizes
take the graded kernel's fidelity. A one-rep smoke on internlm2-1_8b (G = 2) read attention 2.37× / 2.30× faster at
2048 / 3900 keys. Grade and rule: the task doc, "B-P02: built, pending its grade".

**Claim.** `buildResident` selects `attention_fa_blk_g6` or `_g7` only when `nH / nKV` is 6 or 7
(`metal/model.go:1507-1519`); `metal/kernels.go:1430-1433` instantiates only those two. Any other dense hd=128 GQA shape (for
example 32 query heads over 8 KV heads, G=4; 40 over 8, G=5; 16 over 2, G=8) runs `attention_fa` with the core-count
split rule, the kernel R17 measured 2.3-3.5x slower per token-attention.

**Why this is not already closed.** The restriction is stated as a scope choice in the comment at
`metal/model.go:1507-1510` ("instantiated, measured and fidelity-gated only for these two") and in
`metal-decode-attn-r17-2026-09-25.md:463-466`. I found no later record that extends it or kills it.

**MLX analog.** `sdpa_vector_2pass_1` reads G from the threadgroup's y extent (`mlx/kernels/sdpa_vector.h:220`:
`gqa_factor = tptg.y`) and the host sets `group_dims(32, gqa_factor, q.shape(2))`
(`mlx/scaled_dot_product_attention.cpp:817`); any G works. MLX does not share K/V loads across the G simdgroups, so
goinfer's form (one K load, G dots) is cheaper per key; only the generality is borrowed.

**Delta.** In `attention_fa_blk<G>` every per-head statement (`metal/kernels.go:1343-1349`, `1224-1228`, `1238-1246`) touches
only head g's state; G is a loop bound. The arithmetic per head is the same instruction sequence at any G, so a g4/g5/g8
instantiation should agree with g7 head for head. Register pressure grows with G (acc[G], qs[G] float4 plus m, l, s, p:
about 12G registers of live state: 84 at G=7, 96 at G=8, before addressing), so g8 may spill; a g8 that does two 4-head passes is the fallback.

**Band (projection).** For an ungraded G at 2048+ keys: the R17 end-to-end figure at the graded shapes, 1.18-1.41x at
depth (`benchmarks.md:47`), is the only recorded end-to-end basis. The attention share the figure depends on was 36-41%
of the token at 3900 on the graded shapes (decomp `:35`); a shape with more KV heads (8 against 2 or 4) has a larger
attention share, so the band is not an upper bound. Band 1.15-1.4x at 2048+ keys; none at 128.

**Bit-identity / fidelity.** Not bit-identical to `attention_fa` or legacy. Per-head equality between blk<G> and blk<7>
is checkable exactly (same inputs, one head). R17's gate was graded at G=6,7; run it once per new G.

**Probe.** Add `attention_fa_blk_g4/g5/g8` to the instantiation list, extend the test at `metal/decode_attn_r17_test.go:42`
(which loops `g6`, `g7`) to assert head-for-head equality against g7, then time in sequence on a synthetic hd=128 config
with the target head counts at 2048 and 3900 keys, S=16 and S in {8, 24, 32}.
**Kill:** in-sequence attention below 1.5x over `attention_fa` at both depths (R17's registered kill line, `red-october.md:2184`).

---

#### B-P03 [P, Major] REVISIT: the 1536-key floor was measured on the kernel it no longer gates

**Premise.** `metal/model.go:2717-2725`: the floor is "where attention_fa (at a properly-sized split count) starts
beating the shipped kernel", measured by a since-deleted `TestAttentionFA_speedProbe` with S sized to 28: 0.98x at 1024
keys, 1.07x at 1536, 1.26x at 3900.

**Why it no longer holds.** `attnFADepthFloor` gates `attnPlanFor` (`metal/model.go:2855`) and `canUseAttnFA` (`:2570`), so it
now gates `attention_fa_blk`, which is 2.3-3.5x faster than `attention_fa` in sequence (R17 doc `:74-76`: 1.770 vs 4.874
ms at 2048, 2.537 vs 8.574 at 3900 on the 1.5B). R17's own precondition says a candidate that engages below the floor
"has to earn it there too" (`red-october.md:2195-2196`), and R17's wiring section records "the depth floor (1536) ...
unchanged" (`metal-decode-attn-r17-2026-09-25.md:465`). The re-measurement was left undone. The crossover between legacy
and blk has moved to some depth below 1024.

**Why this is not already closed.** Checked the floor comment, R17's precondition 2 and wiring note, and the R17 doc's
e2e depth list (128/2048/3900 only, `metal-decode-attn-r17-2026-09-25.md:505-526`). No record measures blk below 2048 keys.

**What runs at 1024-1535 keys today.** The legacy `attention` kernel. Its cost, from recorded points on the 1.5B:
0.67 ms at 128 keys (decomp `:35`); the 2026-09-18 depth curve, before R2 (so legacy), 128/512/1024/2048 keys =
73.7/69.1/61.9/50.7 tok/s (`metal-depth-r2-2026-09-18.md:55-58`) = 13.57/14.47/16.16/19.72 ms per token, i.e. +0.90,
+2.59, +6.15 ms over the 128-key token. Interpolating to 1535 keys gives about +4.4 ms, so legacy attention is about
3.3 ms at 1024 and 5.0 ms at 1535 (the GEMV share of those curves has since changed; I found no record of a later change to the legacy kernel, but did not diff it).

**blk at those depths (projection).** One trip per simdgroup up to 2048 keys at S=16 (see B-P08), so about 1.77 ms,
flat (the recorded 2048-key point, R17 doc `:76`). The flat assumption is the weak link; a lower fixed cost below 2048 is
likely, a higher one is not supported by any record.

**Band (projection).** At 1024 keys: 3.3 -> 1.8 ms = -1.5 ms of about 13.8 ms (11%). At 1535: 5.0 -> 1.8 = -3.2 ms of
about 15.6 ms (20%). At 512: legacy about 1.6 ms against 1.8, no gain. Band 5-20% of the 1.5B token at 1024-1535 keys
(5% is the case where legacy is cheaper than the curve implies); none at 128 or above 1536. The 7B reads the same
direction (the 7B's `attention_fa` time is 2.7x the 1.5B's at 2048 keys, decomp `:35`: 13.47 vs 4.98 ms). Visible as a 1535 -> 1536 step
in tok/s: projection about 64 -> 82 tok/s on the 1.5B.

**MLX analog.** None for a floor: `sdpa_vector` runs from N=1 up, with the 2-pass switch at N>=1024 on arch 's'/'d' or
N>=4096 with GQA (`mlx/scaled_dot_product_attention.cpp:1565-1567`). The relevant lesson is that the crossover belongs to
the kernel pair being switched, and R17 changed one of the pair.

**Bit-identity / fidelity.** Moving the floor changes the kernel at depths 256-1535, so logits there change bitwise
relative to today (legacy is exact, blk reassociates). The R2/R17 gate was run at K=3900; a lower depth has fewer
reassociated terms, but the gate has to be run at the chosen floor. The snapshot golden straddles the floor on purpose
(`metal/snapshot_golden_test.go:70-72`: "1534 declines, 1535 engages") and would be re-baked (see audit G-04).

**Probe.** Add a test-only `attnFAFloorOverride` read inside `attnPlanFor` and `canUseAttnFA` (next to the existing
`attnFASplitOverride`, `model.go:~2587`, so the grid and the plan agree), then `TestMetalDecodeDecomp`-style
interleaved arms (legacy vs blk, floor 0 vs 1536) at nKeys in {256, 384, 512, 768, 1024, 1280, 1535}, 5 reps x 20
tokens, in-sequence attention = full minus no-op, on the 1.5B and the 7B.
**Kill:** blk not at least 1.05x faster than legacy at 768 keys on either model (floor stays); otherwise set the floor
at the first depth where blk/legacy >= 1.10 and re-run the fidelity gate there.

---

#### B-P04 [P, B, Major] SA-family lane mapping leaves the last trip half-idle at K=1536 and K=3584

**Status, 2026-10-03 (branch `metal-audit`): killed and reverted.** Graded on `TestR18InSequence`: pre-bp04 ÷
production **0.955×** on the 1.5B at depth 128 (0.911–0.917× on the 7B), against the pre-registered 1.02 ship line. The
idle tail T1.4 measured is real, but closing it with lane-pair splitting and shuffles costs more than it saves.

**Counted (code shape).** `sa_rows_acc` loops `for (uint g=lane; g<G; g+=32u)` with `G = K>>5u`
(`metal/kernels.go:558,561`); a lane owns whole 32-weight groups, 32 lanes cover 1024 weights per trip. The SA family
serves qkv, o and gate/up, whose K is the hidden size H.
- 1.5B, K=1536: G=48. Lanes 0-15 make 2 trips (g, g+32), lanes 16-31 make 1. A simdgroup issues 2 trips for 1.5 trips of
  work: 33% extra issue slots.
- 7B, K=3584: G=112. 4 trips for 3.5: 14% extra.
- Down (coal kernel, lane owns a 4-byte word: K=8960 gives 1120 words = 35 trips, K=18944 gives 2368 = 74): balanced, no
  tail. (Not part of this entry.)

**Evidence it is visible in the records.** Gate/up after R18b, 1.5B: 4.064 -> 3.291 ms at depth 128 for 28 layers x
15.5 MB = 434 MB, so 132 GB/s; 7B: 15.336 -> 12.849 ms for 2139 MB, 166 GB/s (R18b doc `:61-66`; bytes from S0 `:48,53`).
The loads-only twin, which shares the lane mapping but has no dependent arithmetic, reads 176 and 187 GB/s (S0 `:48,53`).
The 7B is at 89% of its twin, the 1.5B at 75%; the ordering matches 14% against 33% tail overhead. R18 says it did not
establish what limits gate/up ("at R = 4 every bit-identical variant ties within noise", `metal-decode-gemv-r18-2026-09-26.md:76-78,170-175`);
the twin cannot show a tail because it has almost no per-lane issue work to repeat.

**MLX analog.** `qmv_fast_impl`: 16 weights per lane (`values_per_thread = pack_factor x packs_per_thread = 8 x 2`,
`mlx/kernels/quantized.h:768-775`), `block_size = 512`, taken only when `K % 512 == 0`
(`mlx/quantized.cpp:488`, `qmv_fast_k_alignment` = 512 for 4-bit). K=1536 and 3584 are 3 and 7 whole blocks, so every
lane is busy in every trip. MLX's group-64 layout puts 4 lanes on one group (`scale_step_per_thread`, `:775`).

**Bit-identity.** MLX's mapping as written reorders the float accumulation, so it is not bit-identical to the shipped
kernels. A bit-identical form exists because the group integer sums are exact (|gf| <= 127 x 15 x 32 = 60,960 < 2^24,
R18b doc `:28-29`) and only the order of the float chain matters. Sketch: lane l owns 16 consecutive weights (half a
group), pairs (2m, 2m+1) share group m of a 512-weight trip; one `simd_shuffle_xor(.,1)` adds the two exact integer
partials (gf and sa); the shipped owner lane for group g is `g mod 32`, so one `simd_shuffle` moves the group's integer
sum to that lane, which does `acc += (gf - 8*sa) * float(scale)` in the shipped order. Each lane's accumulator chain is
then the shipped chain and `simd_sum(acc)` is untouched. Cost about (R+1) xor shuffles plus (R+1) index shuffles per
512 weights per row set: below 0.01 op per weight. Load instructions per weight double (uint2 instead of uint4), from
0.0625 to 0.125, against about 3.8 ALU ops per weight. K that is not a multiple of 512 (the 0.5B's H=896 is 1.75 trips) keeps the
shipped mapping; MLX does the same, falling back to `qmv` with a tail.

**Band (projection).** Basis: if the 1.5B gate/up reached the 7B's 166 GB/s, 434 MB / 166 = 2.61 ms against 3.29, -0.68
ms = 6.5% of the 10.46 ms token (R18b doc `:90`); reaching 150 GB/s gives -0.4 ms = 3.8%. qkv and o add a little (K=1536
too). 7B: gate/up already at 89% of its twin, so 1-3%. Band 3-7% (1.5B), 1-3% (7B). Lower than the tail count suggests
because the kernel is partly bandwidth-bound.

**Probe (cheap first).** With the shipped `gemv_w4a8_sa_rows4` at fixed weight bytes (about 15 MB), compare time per
byte at K=1536 (G=48, tail), K=2048 (G=64, two full trips) and K=3072 (G=96, three full trips) by adjusting N. A
time-per-byte advantage of K=2048 over K=1536 of 10% or more confirms the tail. Then prototype the lane-balanced kernel
in `metal/gemv_r18_test.go` (which hard-codes TG 256 at `:615-692`), check byte equality against production through
`TestR18InSequence` (0 differ, as in R18b), and grade in sequence on the 1.5B (the weaker model sets the grade).
**Kill:** below 1.02x in-sequence work on the 1.5B (the owner's standing bar, `red-october.md:2340-2344`: park only a
couple of percent), or any byte difference.

---

#### B-P05 [P, Major] REVISIT of M-10: the down projection after R18

**Premise recorded.** M-10 (`docs/audit-metal-2026-09-12.md:533-564`) measured the shipped one-row SA body staged for the
down shape against the shipped coal kernel and found it 2-6% slower end to end on the 1.5B (`:553-564`), at R=1 and with
the pre-R18 SA arithmetic.

**Why it no longer holds.** Since then the SA family became 1.38x (1.5B) and 1.46x (7B) faster on GEMV work at depth 128
(R18 plus R18b, wired production against pre-R18, R18b doc `:89`), while the coal down projection got only R18's stage-and-rows form (1.20x on the
1.5B, 1.47x on the 7B at depth 128, R18 doc `:123-130`) and nothing from R18b: its pre-registration says "The down
projection is not in scope: it is coal-family and stays as it is" (`red-october.md:2373`), and the R18b candidate keeps
"the down projection unchanged at R18's staged 4" (R18b doc `:51`). The comparison M-10 made (SA vs coal on the down
shape) has not been repeated against the current pair.

**Why this is not already closed.** Checked M-10's closure text, R18 pre-registration (`red-october.md:2300-2312`, "step
2: stage the down projection, keeping coal's per-word accumulation order"), and the R18b scope line.

**Where down stands (counted from recorded figures).** Down is the slowest int4 GEMV after R18b:
1.5B: pre-R18 3.06 ms in sequence (decomp `:37`) / 1.20x (R18) = 2.55 ms for 28 x 7.7 MB = 215 MB, 84 GB/s.
7B: 14.19 ms / 1.47 = 9.65 ms for 1070 MB, 111 GB/s. Gate/up after R18b: 132 and 166 GB/s.
The coal body costs: one 4-byte weight load plus one 2-byte scale load per 8 weights per row (0.25 load instructions per
weight against SA's 0.0625), and one f32 FMA per 8 weights against one per 32
(`metal/kernels.go:622-648`, `W4A8_BODY` `:253-266`; M-10 `:545-547`). Dispatch is `H*32/R` threads at TG 256
(`metal/model.go:2634-2636`): 48 threadgroups on the 1.5B, 112 on the 7B, a single wave.

**MLX analog.** One inner loop for every projection: `qmv_fast_impl`, `mlx/kernels/quantized.h:756-821`. (For Qwen2.5-1.5B
the down K=8960 is not a multiple of 512, so MLX itself takes the slower `qmv` with a tail there, `mlx/quantized.cpp:488`.)
The borrow is the idea of one fast form for all shapes; the MLX body itself is not what goinfer needs.

**Why bit-identity is the real cost.** The down kernel's per-word float chain is pinned by the snapshot golden and by
the batched-verify down kernel, which reproduces production's order on purpose (MC3 record
`concurrency-mc3-s0-2026-09-26.md:80`, "production's down kernel sums per 8-k word, not per group"). Switching decode's
down to the SA body changes bits (M-10 `:545-547` said so) and breaks decode/verify agreement unless
`mc3_mma_btd` moves too. That coupling belongs to area E.

**Band (projection).** Basis: if down reached gate/up's rate on each model: 1.5B 215 MB / 132 GB/s = 1.63 ms against
2.55, -0.92 ms = 8.8% of 10.46 ms; 7B 1070 MB / 166 = 6.45 against 9.65, -3.2 ms = 9.7% of 33.1 ms. Band 0-9% (1.5B),
0-10% (7B); the central expectation is lower because down's K is 5.8x and 5.3x longer than H and its output is only H
rows. The 7B cannot stage K x 2 B as half (18944 x 2 = 37,888 B exceeds the 32 KiB threadgroup budget), so the 7B needs
int8 staging or a two-chunk K loop.

**Probe.** In `TestR18InSequence`, dispatch the existing production `gemv_w4a8_sa_resid_rows4` (`metal/model.go:1479`,
the o-proj kernel with the residual epilogue) at the 1.5B down shape (N=1536, K=8960, 17.9 KB staging fits) against
`gemv_w4a8_resid_staged4`, same dispatch site, interleaved, 7 reps. Check byte difference (expected non-zero, size it
with the teacher-forced gate). **Kill:** below 1.03x in-sequence down work on the 1.5B.

---

#### B-P06 [P, B, Minor] Threadgroup geometry of the rows kernels is untested

**Counted.** Every R18/R18b dispatch is `DispatchTG(..., 256, ...)` (`metal/model.go:2609` gate/up, `:2440` down, `:2653`
qkv, `:2762` o): 8 simdgroups per threadgroup, `row0 = (tgid*8 + sgid)*R` (`metal/kernels.go:559`), and `gemvRowsFor` tiles rows by
8R (`metal/model.go:2650-2657`). Staging costs one K x 2 B write plus a barrier before any weight load is issued
(`metal/kernels.go:556-557`). Grid sizes at R=4: 1.5B gate/up 560 TGs, qkv 64, down 48; 7B qkv 144, down 112.

**Record.** R18 lists the open items itself: "Occupancy and latency hiding are therefore the natural next suspect, but
nothing here measured them. Candidates for a later look are a per-shape R rule, TG sizes other than 256, and more rows
per threadgroup at fewer simdgroups" (`metal-decode-gemv-r18-2026-09-26.md:170-175`). R18b changed the inner loop and
left the TG size unchanged (`metal-decode-gemv-r18b-2026-09-26.md:21`), so the best TG size for the new loop is unknown.
The record says "gate/up is capped at ~1.24-1.28x whatever the arithmetic, the activation source, R or the unrolling"
(`:76`).

**MLX analog.** `group_dims(bk, 2, 1)` = 64 threads, 2 simdgroups x 4 rows = 8 rows per TG (`mlx/quantized.cpp:495`,
`mlx/kernels/quantized.h:769-770`), no threadgroup memory and no barrier; x is read per lane from device (16 halves).
MLX itself treats rows per simdgroup as a per-device tuning axis (`mlx/quantized.cpp:488-495`, 2 instead of 4 on M5 Max
for nvfp4). Cost of copying MLX's shape: the staging prologue is amortised over 8R rows per TG; at 64 threads it is over
2R, so activation L2 traffic rises 4x (1.5B gate/up: 0.86 MB to 3.4 MB, 5.5% to 22% of the weight bytes), which is why
the sweep has to include staged and unstaged variants. R18b's first cut lost in sequence for exactly an activation re-read
reason (R18b doc `:32-42`).

**Band (projection).** Only gate/up has a recorded gap to its twin on the 1.5B (132 vs 176 GB/s). If geometry closed
half of it: -0.4 ms = 3.8% of the token; all of it: 6.6%. B-P04 and this entry overlap on the same gap; I give the
larger share to B-P04 because the count there is concrete. Band 0-6%, likely <= 3%.

**Bit-identity.** Exact: TG size changes only the row-to-simdgroup map; the lane order and per-row chain are untouched.
**Probe.** TG in {64, 128, 256, 512} x R in {2, 4} x {staged half, device-read} at the 1.5B and 7B gate/up shapes,
standalone first, then in sequence. **Kill:** below 1.03x in sequence on the 1.5B.

---

#### B-P07 [P, Minor] blk's QK stage: 32 simd_sums per head per block

**Counted.** Per 32-key block per simdgroup, `for j<32 { for g<G { s[g] = (lane==j) ? simd_sum(dot(qs[g],k4)) : s[g] } }`
(`metal/kernels.go:1357-1362`): 32 x G simd_sums. If `simd_sum` lowers to 5 shuffle+add steps (the tree is the xor
butterfly 1,2,4,8,16, MC3 record `:74`; the lowering is not recorded) that is 320 cross-lane ops per head per block.
A reduce-scatter that halves the key set at each xor level (levels 1,2,4,8,16 in that order) costs 16+8+4+2+1 = 31
shuffles + 31 adds + about 62 selects = 124 ops, 2.6x fewer. Per key at G=7: about 70 of about 157 ops per key per
simdgroup (QK: 1 load + 4 cvt + 28 FMA + 70 simd_sum ops + 7 selects; PV: 1 load + 4 cvt + 7 shuffles + 28 FMA; softmax
about 7 amortised) become about 27, -27% of the per-key instruction count.

**Bit-identity.** The reduce-scatter pairs lanes at the same distances in the same order as the measured tree
(`concurrency-mc3-s0-2026-09-26.md:74`: tree measured on this GPU as exactly the xor butterfly 1,2,4,8,16, 0 differences in
65,536 random vectors; the 16,8,4,2,1 order differed on 62%). At each level a lane and its partner add the same two
partial values (commutative, so identical bits on both lanes), and the key a lane retains after level 5 is its own index,
so the retained value is the full butterfly sum for that key. Scores are bit-identical to the shipped kernel where
that tree holds; MC3 already uses the same technique for batched GEMM outputs (`mc3_mma_bt`) and gates it with a build-time
self-check (`:108-113`). A different Apple generation could differ, so the kernel needs the same gate.

**MLX analog.** None. MLX does one `simd_sum` per key per head (`mlx/kernels/sdpa_vector.h:119,278,410`).

**Band (projection).** Basis: attention is 17% of the 1.5B token at 3900 (2.537 of 14.82 ms, R17 doc `:76,83`) and 13.5% of
the 7B's (7.072 of 52.3); if attention time followed the instruction count, -27% of it is 4.6% (1.5B) and 3.6% (7B). Whether
the kernel is issue-bound is not known statically (its effective KV rate is 44 and 32 GB/s by count, below), so the band
is 0-4.6%. It does not apply below 1536 keys.

KV bytes count for the effective rate: 1.5B, 28 layers x 2 KV heads x 128 x 2 B x 2 (K,V) x 3900 keys = 111.8 MB in
2.537 ms = 44 GB/s; 7B, 4 KV heads, 223.6 MB in 7.072 ms = 32 GB/s, against a 178-182 GB/s ceiling.

**Probe.** (1) Bit-equality: prototype reduce-scatter in a test kernel next to `TestMC3SimdSumTree`, 65,536 random
vectors, 0 differences required. (2) In sequence, 1.5B and 7B at 3900 keys, interleaved arms, 5 reps x 20 tokens.
**Kill:** any difference in (1); below 1.03x attention in (2). Any edit to the blk region trips
`TestAttnFABlkIsTheGradedKernel` (SHA-256 of the region): the new kernel must first be shown bit-identical to the graded
one, and the hash updated in the same change.

---

#### B-P08 [P, Minor] blk's fixed split gives a cost staircase in depth

**Status, 2026-10-03: stands on the 7B, parked on the 1.5B (T1.7).** Attention per key max/min 1.68–1.71× on both;
the step across a trip boundary is 4.6–4.8× the median 128-key step on the 7B and 2.5× on the 1.5B (the pre-registered
staircase reading, beside the per-key one). Record: the task doc, "The night of 2026-10-02: results".
**PARKED on both models by the owner, 2026-10-03; not planned.** Low value: the projected cost is 0 to 5.5% of a token at a trip boundary and about 2.5% on average, zero at the graded
depths (2048 and 3900), and the probe found the staircase on the 7B only. It would reopen only if a workload actually sits near a boundary (2049, 4097, ... keys) on the 7B, or if the block
kernel is being changed for another reason. Any remedy is a block-size or simdgroup-count change inside the threadgroup (S alone does not remove it: R17 `:76`), and goes through the
set-B fidelity gate and a pre-registered night A/B on the Mac; nothing was changed.

**Counted.** Per threadgroup, simdgroup `s` takes blocks `chunkStart + s*32 + 128*i` (`metal/kernels.go:1352`), with
`chunkLen = ceil(nWin / nSplit)` (`:1201`) and nSplit = 16 (`metal/model.go:2705`). The busiest simdgroup (sg0) runs
`ceil(chunkLen / 128)` trips, which is 1 for n <= 2048, 2 for 2049-4096, 3 for 4097-6144. Both graded depths are
balanced points: 2048 (chunk 128 = 4 blocks, 1 per simdgroup), 3900 (chunk 244 = 7.6 blocks, 2 per simdgroup).

**Record.** The R17 table has two points, 1.770 ms (2048) and 2.537 ms (3900) on the 1.5B, 4.898 and 7.072 on the 7B
(`metal-decode-attn-r17-2026-09-25.md:76`), and says the response to S is not monotone so "a production rule would need
its own sweep" (`:81-83`; `metal/model.go:2705-2720` comment). Non-monotonicity fits the trip count: S=24 at 3900 gives
chunk 163 = 5.1 blocks over 4 simdgroups = 2 trips for 1.3 trips of work; S=32 at 2048 gives chunk 64 = 2 blocks, 2
simdgroups idle (2.497 vs 1.770 ms).

**Band (projection).** Linear model from the two recorded points (cost = a + b x trips): per trip 0.77 ms (1.5B), 2.17 ms
(7B). One extra key at 2049 costs a whole trip: +0.77 ms (5.5% of a 14 ms token) and +2.17 ms (4.8% of 45 ms); the mean
waste over a step is half that. Band 0-5.5% at depth, mean about 2.5%, zero at the graded depths.

**Probe.** Measure, do not fix first: in-sequence attention at nKeys = 1536..6144 in 128-key steps, both models. **Kill:**
max/min of per-key cost over the sweep below 1.15x (no sawtooth worth a rule). If it is real, the remedy is a block-size
or simdgroup-count change inside the threadgroup (S alone does not remove it: R17 `:76`), and any such change goes through
the set-B fidelity gate.

---

#### B-P09 [P, Minor] `rmsnorm_quant` barrier and pass count

**Counted.** `metal/kernels.go:24-64`: pass 1 reads x for the sum of squares; pass 2 re-reads x and w for the amax; pass 3
re-reads both to quantise. Barriers: 1 after `red[tid]`, 8 in the `s = tgs/2 .. 1` tree, 1 after `rsqrt`, 2 for the
simdgroup-max exchange = 12. The last five tree levels (`s` = 16..1) run entirely inside simdgroup 0.

**Bit-identity.** The sum-of-squares tree is a pinned contract (`metal/kernels.go:34-38` comment, `tgReduceNorm` = 256). Keeping
the same operand pairing for levels s=128, 64, 32 through threadgroup memory and doing s=16..1 with `simd_shuffle_down`
in the same order is bit-identical (the pairing `red[tid] += red[tid+s]` for tid < s is a shuffle-down). Holding x and w
in registers (6 floats per thread at H=1536, 14 at H=3584) removes passes 2 and 3's device reads and changes no sum.

**MLX analog.** `rms_single_row`: x in a register array, one load, `simd_sum` plus one cross-simdgroup exchange
(`mlx/kernels/rms_norm.metal:13-80`). MLX's reduction order differs from goinfer's pinned tree, so copying it as written
would change bits; the register caching alone would not.

**Band (projection).** The category is 0.49 ms of the 1.5B token (decomp `:41`, 57 dispatches = about 8.6 us each, work
only, floor excluded). Halving per-dispatch time gives -0.25 ms = 2.3% of the 10.46 ms token; the lower end is 0.3%.
Possibly already attempted: the kernel comment cites autoresearch rounds for the vectorised amax load
(`metal/kernels.go:39-45`) and the results ledger `scripts/autoresearch_rmsnorm_results.tsv` is not in the snapshot (see f).

**Probe.** Register-cached variant, byte-equal against shipped on the snapshot golden inputs, in-sequence
norm+quant category (decomp method). **Kill:** below 1.01x full token on the 1.5B at 128.

---

#### B-N01 [N, Minor] Metal cannot honour `-embed-int4`

**Claim.** The Metal head load requires int8: `int8BufA` falls to `int8Buf`, which returns "weight kind ... is not int8"
for anything else (`metal/model.go:1358-1365`, `:444-450`). `decoder/weightmat.go:96-106` offers an opt-in int4 head
(`EmbedInt4`, "Lossy and opt-in — a 1.5B Q4_K_M spike measured ~2.3 pts top-1"), and `docs/giw-bundles.md:24` documents
`-embed-int4`. On Metal such a bundle fails at residency build (what happens next is area C's).

**Weight.** Head bytes per token: 1.5B (tied) 234 MB int8 against about 131 MB int4 (0.5625 B/weight); 7B 545.6 MB against
about 307 MB (S0 `:50,55` give the int8 sizes). Saving 102 MB (1.5B, at the head's 161 GB/s: 0.63 ms = 6% of 10.46 ms) and
238 MB (7B: 1.46 ms = 4.4% of 33.1 ms). This is a lossy trade the repo has pinned off by default and records as
deliberate (`decoder/weightmat.go:78-94`, N-40 in the Sep 12 audit); I do not propose changing the default. The point is only that
MLX's 4-bit checkpoints carry a 4-bit head, so the byte share of the MLX gap (about 5.6% of the 7B token) is a chosen
precision difference, and that the opt-in has no Metal kernel.
**Consolidation note.** `internal/loadflags/loadflags.go:130-139` already documents this gap and keeps the Metal default off; whether Metal should take an
int4 head is owner decision O3 (§10), not a defect.
**Probe, if wanted.** The int4 head would use the existing SA rows kernel at N=V (V % 8 == 0 for the Qwen vocab 151,936);
gate with the teacher-forced top-1 check that produced the 2.3-point figure.

---

#### B-D01 [D, Minor] "AHEAD of Ollama in every cell" covers six cells

**Status, 2026-10-03: the 0.5B cells measured (T1.11).** Same session on main: goinfer ÷ Ollama **0.804×** at 2048 and
**0.568×** at 3900 (1.143× at 128). The row's claim was already scoped to the six 1.5B and 7B cells (T0.5); it now
carries these 0.5B cells.

`docs/benchmarks.md:47` ends the Metal decode row with "goinfer is AHEAD of Ollama in every cell" after R18b, then lists the
1.5B and 7B at 128/2048/3900. The R18b e2e table has only those six cells (`metal-decode-gemv-r18b-2026-09-26.md:104-113`).
The same row's pre-registered 0.5B cells read 0.75x (2048) and 0.58x (3900) and have no later number, because the 0.5B
does not reach the block kernel (B-P01) and the R17/R18/R18b e2e runs did not include it. Reword to "every 1.5B and 7B
cell" or add the 0.5B cells.

#### B-D02 [D, Minor] Dispatch-count comment omits the combine

`metal/model.go:2930-2934` (the N-10 fix) counts 7 attention-block dispatches in "the baseline dense case". At fa depths
the combine kernel is an eighth. Per layer: 7 + 4 (norm+quant, gate|up, swiglu+quant, down+residual) = 11, 12 with the
combine; per token 28 x 12 + final norm + head = about 338 above the floor, 28 x 11 + 2 = 310 below it (the 310 in the
Sep 12 M-16 entry is the below-floor count). One-line comment fix.

---

### (d) Carry-forward (Sep 12 audit, `docs/audit-metal-2026-09-12.md`)

| ID | Status | Evidence |
|---|---|---|
| M-09 (decode attention K read, staged corner) | NEGATIVE-CLOSED | Probe `metal/attn_kread_staged_probe_test.go` exists; staged 39.6 ms vs shipped 17.0 ms (0.43x) at 2048 keys (audit `:514-521`). Two loose ends: its maxAbs 1.7e38 correctness mismatch was never root-caused (`:519-531`), and the premise (a 32-lane 512 B-strided K gather) no longer describes the production decode attention at hd=128, G=6/7, >=1536 keys: blk loads each K row coalesced, 32 lanes x half4 = 256 B (`metal/kernels.go:1357-1360`). It still describes the legacy kernel, which serves <1536 keys, hd=64, kvI8, windows and sinks (B-P01, B-P03). |
| M-10 (down projection unstaged) | NEGATIVE-CLOSED, premise stale | Reverted `model.go` change measured 2-6% slower at the time (audit `:553-564`); coal down is now staged at R=4 (`metal/model.go:2634-2636`), the int4 GEMV work is 1.38-1.46x faster at depth 128 than when M-10 measured it (R18b doc `:89`). REVISIT filed as B-P05. |
| M-16 (hazard tracking / serial encoder) | NEGATIVE-CLOSED | Untracked 99.6-101.5% of tracked in a 310-dispatch probe (audit `:906-918`; the probe is in aikit, not in this snapshot). Decode graph is nearly a chain (12 dispatches per layer, few independent siblings after QKV and gate|up fusion), so MLX's concurrent dispatch type has little to overlap here. |
| N-03 (§B3 depth curve measures a path serve never takes) | CLOSED-VERIFIED (superseded) | Superseded 2026-09-18 by `bench_peer.py` over the real server (audit `:1274-1280`); decode rows now come from the served path (`benchmarks.md:47`). |
| N-09 (softcap in greedy decode) | CLOSED-VERIFIED | `finalizeLogits` is called after every logits readback and applies the softcap host-side: `metal/model.go:1856,1855-1862`. |
| N-10 (stale dispatch counts) | PARTIAL | Fixed 2026-09-13; the replacement count at `metal/model.go:2930-2934` omits the fa combine (B-D02). The 310 figure is correct below the floor. |
| N-13 (snapshot golden names `attention_f32`) | CLOSED-VERIFIED | `metal/snapshot_golden_test.go:75-79` explains the removal (N-28). `attention_f32` is built only under `r.kvF32`, which is hard-wired false. |
| N-17 (kernels with no production pipeline) | CLOSED-VERIFIED | The six listed kernels have no non-test pipeline reference at HEAD (grep of `"gemv_w4a8_bias"`, `_sa_amax`, `_sa_bk`, `_sa_qv`, `"gemv_w8a8"`, `"rope2_kv"`: tests only); header at `metal/kernels.go:5-17`. `gemv_w4a8_sa_amax` is still the one with no reference at all, kept by decision. |
| N-30 (swiglu_quant double evaluation, rope2_kv) | NEGATIVE-CLOSED | Recorded as a trade needing a device scratch buffer (audit `:1509-1520`); `rope2_kv` unwired (0.6%, `metal/kernels.go:5-17`). The stale "K<=1536" comment now corrects itself at `metal/kernels.go:399`. Not re-proposed. |
| N-37 (micro-bench shape is cache-resident) | OPEN as a caveat, mitigated | `gemv_w4a8_coal_bench_test.go` exists; R18 and R18b grade in sequence through `TestR18InSequence` and explicitly cite the Stage-B lesson (R18b doc `:40-42`). |
| N-40 (tied head runs int8, "4-bit both sides") | CLOSED-VERIFIED (labelling) | Head pinned int8 by decision: `decoder/weightmat.go:78-94`; head kernel at 161-163 GB/s, 90% of ceiling (S0 `:50,55`). See B-N01 for the opt-in. |

---

### (e) Checked and found correct

- **Fusion state against MLX.** goinfer decode per layer, with the dispatch each fusion removes: norm+quant in one kernel
  (`rmsnorm_quant`, `metal/kernels.go:24`); QKV in one GEMV with bias (`metal/model.go:2964-2965`); Q and K RoPE in one dispatch
  (`rope2`, `metal/kernels.go:886`); gate|up in one GEMV (`metal/model.go:2607-2609`); swiglu+quant in one kernel
  (`metal/kernels.go:1859-1899`); residual in the o and down epilogues (`gemv_w4a8_sa_resid_rows`, `gemv_w4a8_resid_staged`).
  12 dispatches per layer at fa depths, 11 below. MLX's quantized kernels have no prologue or epilogue fusion
  (`mlx/kernels/quantized.h:756-821` writes y only), no norm-into-GEMV, no residual-into-GEMV, no swiglu-into-down, no
  rope-into-attention; MLX's per-layer graph is bias add, rope, cache update and residual add as separate ops. MLX has no
  decode-path fusion that goinfer lacks (its compiled elementwise swiglu is covered by `swiglu_quant`). Tried and unwired in goinfer:
  rope2+kv_store (0.6%, `rope2_kv`), quant-into-GEMV (about 0.97x), recorded in audit §5 `:1633-1650`.
- **Per-weight work, R18b against MLX.** goinfer `sa_rows_acc` (`metal/kernels.go:561-578`) per weight at R=4: per half-word
  (4 weights per row) one select/shift/mask, 4 and-masks, 4 int-to-float converts, 4 FMA = 13 ops, 3.25 per weight; the
  Σa dot (4 converts + 4 FMA, shared over R=4 rows) 0.5 per weight; group finalize about 0.09. Total about 3.8-3.9.
  MLX `qdot` (`mlx/kernels/quantized.h:191-250`, 4-bit branch): per uint16, 4 and-masks, 4 converts, 4 FMA = 3.0 per
  weight; `load_vector` (`:28-70`) 16 converts, 16 adds, 12 pre-scale multiplies per 16 values over 4 rows = 0.69 per weight;
  finalize `scale*accum + sum*bias` 0.19. Total about 3.9. Parity. Load instructions per weight: goinfer 2 per 32 weights
  per row (uint4 + half scale) plus 8 threadgroup half4 reads per group shared over R = 0.125; MLX 4 uint16 weight loads
  (1 if merged) + scale + bias per 16 weights = 0.19-0.375 plus the x reads: 1.7-3.3x more. The merge is not knowable
  statically (f). Conclusion: MLX's form does not attack the per-weight issue term beyond what R18b took.
- **R18/R18b bit-identity argument.** |a| <= 127, nibble <= 15, 32 weights: |group integer sum| <= 60,960 < 2^24. The
  pre-scaled half `a * 16^-(k mod 4)` is exact (127 x 2^-12 is a normal half), the masked field `n * 16^k` is an exact
  float, their product is the exact integer `n*a`, every partial sum is an exact integer, and the per-row chain
  `acc += (gf - 8*sa) * float(scale)` is in shipped lane order (`metal/kernels.go:576`). Verified in the record by 0 differences
  through the executor at three depths on both models and 10/10 snapshot golden (R18b doc `:84-90`).
- **MLX layout against goinfer's.** MLX keeps weights, scales and biases as three separate row-major arrays (`qmv_fast_impl`
  pointers, `mlx/kernels/quantized.h:790-792`) at group 64: 0.5 + 4/64 = 0.5625 B/weight. goinfer keeps weights and half
  scales at group 32: 0.5 + 2/32 = 0.5625 B/weight. Same bytes. Per row per trip goinfer reads 512 B of weights (4 whole
  128-B lines) and 64 B of scales; MLX reads 256 B of weights and 16 B each of scales and biases. No extra transactions
  either way. No layout lesson from MLX.
- **fp16 against fp32 accumulation.** MLX accumulates in float with half in/out (`U = float`, `quantized.h:28`
  `load_vector<T, U>`; `sdpa_vector.h` `typedef float U`). goinfer accumulates in f32 everywhere with exact integer inner sums
  for W4A8; the one half-accumulating path was the R1 W4F16 lane, killed at 1.03x (`red-october.md` R1). Bit-identity
  consequence: MLX's K-block interleave and register accumulation would change bits; B-P04 shows the one MLX mapping
  that can be kept bit-identical.
- **Attention partitioning against MLX.** 1-pass (`sdpa_vector`): one 1024-thread TG per query head, keys interleaved over
  32 simdgroups, no GQA sharing. 2-pass: grid (kv heads, batch, blocks), TG `(32, G, q_len)`, key `i = block; i += blocks`,
  one `simd_sum` and two `exp` per key per head (`sdpa_vector.h:179-323`). Block count: arch 's' 64, and 128/256/512/1024
  by N when n_simds > 4; 'd' 128/256/512/1024; other archs 64 if n_simds >= 4 else 32 (`scaled_dot_product_attention.cpp:776-806`).
  Switch to 2-pass: arch 'd' or 's' and N >= 1024, or GQA and N >= 4096 (`:1565-1567`). The duplication-free `_gqa` form is
  for G in {8, 12, 16} at N >= 8192 only (`:756-761`). goinfer: 16 splits x 4 simdgroups = 64 partitions per KV head,
  1.5B grid 32 TGs, 7B 64 TGs, with K loaded once per key for all G heads. At 64 per KV head it equals MLX's 'other'
  setting; for arch 's' at N in 1024-8192 with G > 4 MLX uses 128, and R17's S sweep (S=24, 32 and 48 gave no gain over 16 beyond noise on this
  kernel, R17 doc `:76`) says goinfer gains nothing from more. `sdpa_blocked.h` is not a decode kernel (scale copy and causal mask
  helpers for the blocked prefill path); the 2-pass decode form is in `sdpa_vector.h`.
- **R17 plan consistency.** `attnPlan` carries `fa`, `nSplit`, `f16Lane`; `attnPlanFor` and `canUseAttnFA` read the same floor
  and `attnFASplitFor`, so a pre-encoded buffer that disagrees with the current key count is re-encoded
  (`metal/model.go:2837-2859,2861-2901`; R17 doc `:333-358` records the original defect and the fix).
- **blk tail and empty blocks.** For the last 32-key block (`nb < 32`), `s[g]` stays -inf outside `nb`, `p` is 0 there,
  `alpha` is guarded for `m == -inf`, and the 4-simdgroup merge guards `-inf` (`metal/kernels.go:1365-1380,1396-1409`). With S=16 and
  nKeys >= 1536 no split is empty (`nKeys/32` cap, `metal/model.go:2888-2906`).
- **LM head.** `gemv_w8a8_coal` at 32-thread TGs, one row per TG (`metal/model.go:1852`): 161 GB/s (1.5B) and 163 (7B) against a
  178-182 ceiling (S0 `:50,55`), byte-minimal for int8. Nothing to gain at fixed bytes.
- **rope.** goinfer `rope2` computes `cos/sin(pos * invf[dd])` per pair (`metal/kernels.go:896`), the same shape as MLX `rope_single`
  (`mlx/kernels/rope.metal:24-26`, `fast::cos/sin`); no table in either.
- **Wiring guards.** `gemvRowsFor` falls back to the shipped kernel where rows do not tile (`metal/model.go:2650-2657`), the down
  rows kernel only where K staging fits (`metal/model.go:1470`), and the SA rows kernels' no-guard tail hazard (audit C-10) is
  covered by `bad8` width checks (`model.go:~1238-1262`).
- **`*_nax*` files** in MLX (M5 only) were not read as findings.

---

### (f) Questions I could not settle statically

1. **Arch letter of an M1 Pro in MLX's switch.** `device.cpp:604-619` maps the last arch character ('g' base/pro, 's' max, 'd'
   ultra per its comments); the 1-pass/2-pass threshold (`scaled_dot_product_attention.cpp:1565`) and block counts
   (`:776-806`) depend on it. Not knowable from source; it only affects the MLX comparison, not a goinfer change.
2. **Lowering of `simd_sum`.** Five shuffle+add steps or a hardware reduction? MC3 measured the tree order, not the
   instruction sequence. B-P07's 2.6x op-count ratio assumes the first.
3. **Register counts and occupancy.** Per-lane registers of `sa_rows_acc` at R=4 and `attention_fa_blk<7>`, resident
   threadgroups per core, and whether the `g` loop is software-pipelined. They decide B-P04's tail cost, B-P06, and whether g8
   spills (B-P02). R18 recorded two collapses it read as register spills (`dvi` 0.32-0.35x, `fm8` 0.15-0.17x, R18 doc `:73-74`), so the cliff exists.
4. **Whether MLX's four `uint16_t` weight loads in `qdot` merge into one 8-byte load.** Moves the load-instruction ratio
   between 1.7x and 3.3x; the ALU parity conclusion does not depend on it.
5. **Legacy attention cost at 512-1535 keys on the current build.** B-P03's basis is the 2026-09-18 end-to-end curve; no
   record measures it after R18/R18b, and blk's cost below 2048 keys is an extrapolation of one point.
6. **Missing from the snapshot:** `docs/completed/metal-verdict.md` (cited by M-09/M-10 and the kernel comments) and
   `scripts/autoresearch_rmsnorm_results.tsv` (the ledger the `rmsnorm_quant` comment points to). B-P09 may repeat an
   attempt recorded there, and the M-09 `:95-99,136-145,175-178` evidence could not be re-read.
7. **mlx-lm's 4-bit head.** Whether the mlx-community checkpoints quantise `lm_head` is a property of the checkpoint
   converter, not of the library source read here; B-N01 and the 5.6% byte share assume they do.
8. **Record against code for the int4 embedding default.** `docs/measurements/mc3-prefill-attr-2026-09-28.md:27` says "since
   9ccf7fb1 the int4-embedding default makes the Metal resident decline to the CPU"; at HEAD `embeddingWith` defaults to the
   int8 pin (`decoder/weightmat.go:96-106`) and I found no int4 default in `internal/serveapp/main.go`. Either the default
   was reverted or the line is stale; area C/D. **Resolved at consolidation:** `internal/loadflags/loadflags.go:130-139` defaults
   `--embed-int4` on except with `-backend metal`, where it stays off so the resident path survives; the 2026-09-28 line describes the
   state before that guard (§3 note 1).
9. **Cross-session MLX figure.** The 0.82x and 0.75x ratios in (a) pair goinfer's R18b e2e numbers (89.3 and 29.4 tok/s,
   `metal-decode-gemv-r18b-2026-09-26.md:108-113`) with MLX's 109.1 and 39.1 from 2026-09-18 (`r12-mlx-row-2026-09-18.md:17-22`,
   `mlx-lm` 0.31.3, a different session and a different quantisation scheme); the box's recorded drift is about 3.5%.
   A same-session three-way run would replace them.
10. **7B gap accounting.** 33.1 ms goinfer against 25.6 ms MLX at depth 128: head bytes about 1.5 ms (238 MB at 160 GB/s),
    down at gate/up's rate about 2.5-3.2 ms, leaving about 4 ms unattributed (attention 0.8, norm/rope/act-quant 1.8-2.6,
    dispatch floor 1.1 in goinfer's own token; MLX's equivalents are unknown).


## 6. Area C — Runtime: executor, command buffers, sync, purego cost, memory, load path, residency, alias, fit accounting

goinfer HEAD 844700f8, aikit v1.51.0, MLX HEAD 9c3d355. Static review: nothing built or timed.
Evidence labels: [R] recorded figure (doc:line), [C] counted from code shape (count shown), [M] MLX source (file:line under mlx/backend/metal/), [D] derived arithmetic on [R]/[C] figures, [P] projection (carries band and basis).

---

### (a) Summary

1. The dense decode runtime is at its structural floor. One command buffer, one serial compute encoder, one commit and one wait per token; 310 dispatches and 1,042 purego transitions per token [C]; encode-ahead already hides about half of the non-GPU time (0.83 to 0.45 ms on the 1.5B, 0.92 to 0.50 ms on the 7B [R] concurrency-mc3-s4-2026-09-27.md:35-36). What host-side work remains is bounded by that 0.45-0.50 ms; the GPU-side dispatch floor (0.86/1.06/0.65 ms for 1.5B/7B/0.5B [R] metal-decode-decomp-2026-09-25.md:43) is not reachable by any host-side batching.
2. The largest item in this area is memory, not speed. C-P01 (Major): the paged-expert f16 scale cache is a second, anonymous copy of scales the v15 .giw already aliases; about 1.43 GB of the M26's 2,967 MB footprint [C], and the largest anonymous (swappable) allocation I could attribute on the load path.
3. C-C01 (Major, conditional): the decoder's fit guards have no knowledge of Metal's context ceiling or its f16 KV, so an unpinned load on a tight machine is auto-pinned to a context 3-7x the 4096 default (recorded in N-41) and, for a model with a window above 32768 on a roomier machine, to a value Metal refuses, which moves the whole forward to CPU.
4. The only structural runtime technique MLX has that goinfer lacks is the async token chain (the next graph is encoded and committed without a host round trip; eval.cpp:29-69). C-B01 prices it: 0.30-0.45 ms/token [P], 2.2-3.4% on the 1.5B, 0.7-1.1% on the 7B. C-P02 is its sampled-decode sibling.
5. MLX questions, in one line each (details in section (c)): MLX wired limit is 0 by default and is a budget for residency sets, not a sysctl; goinfer already has the bindings and has a recorded reason not to pin dense weights; concurrent dispatch gains about nothing on the dense chain (11 strictly chained dispatches per layer); setBytes, argument buffers and fewer encoders do not reduce transitions.
6. Recorded negatives: ICB, unretained references, megakernel and dispatch-count fusion premises hold on today's kernels (bounds recomputed). The shared-event premise was measured on a dense shape and its target cost has since changed on the paged shape; the re-run was never done. MLX's spin fence is a different mechanism from the MTLSharedEvent that was tested (C-B03, REVISIT).
7. Sep 12 carry-forward in this area: M-01, M-07 (row4 half), M-12, M-13, M-14, C-05 closed and verified in code; M-16 negative stands but is limited to serial encoders; M-11/G-05 still open; N-41 live in a new form (C-C01).
8. Not settleable statically: the GPU-idle share of the 0.45 ms, whether untouched KV pages count at allocation, MLX's M1 Pro per-chip bucket, scale-page behaviour under pressure on the M26 (section (f)).

---

### (b) Findings table (ordered by effect)

| ID | Sev | Claim | Evidence | MLX analog | Band | Probe |
|---|---|---|---|---|---|---|
| C-P01 | Major | Paged-expert f16 scale cache holds ~1.43 GB of anonymous heap that duplicates scales the v15 .giw aliases; guard does not price it | the paged branches of metal/gemma4_moe.go and metal/moe.go and `int4DirectBytes` in metal/model.go, at `844700f8` (removed 2026-10-02); s6-alias-2026-09-24.md:262-270 | none (MLX has no paging) | -1.43 GB of 2,967 MB (-48%) [C]; decode -6% to 0% [P] | heap profile + footprint A/B on M26 v15; kill if reduction < 1.0 GB or decode worse than -3% |
| C-C01 | Major (conditional) | Decoder guards price f32 KV and have no Metal ceiling; auto-pin inflates Metal KV vs the 4096 default or exceeds 32768 and forces CPU fallback | decoder/fitguard.go:366-423,742-757; decoder/model.go:601-611; metal/model.go:22-60; decoder/fitplan.go:233-243; decoder/residency.go:1149-1168; audit N-41 (:1588-1603) | resident.cpp / allocator.cpp budget by memory limit, not by kernel ceiling | KV 3-7x default at pin [R]; CPU fallback = whole-forward loss | unit test with injected hostRAMAvailable and a 131072-window config; kill if guardGIWFit never returns > 32768 |
| C-B01 | Minor (Major only at top of band on 0.5B) | No on-device token feedback: every token pays wake + 608 KB copy + host argmax + embed row + commit between kernels | metal/model.go:2091-2156,2217-2247; decoder/model.go:2264-2321 | eval.cpp:29-69 async_eval; device.cpp:511-513 | 0.30-0.45 ms/token [P]: 2.2-3.4% (1.5B), 0.7-1.1% (7B), 5.3-7.9% (0.5B, assumes the 1.5B gap) | log kernStart(t+1)-kernEnd(t) (aikit metal.go:808-814); kill if gap < 0.15 ms |
| C-P02 | Minor | ForwardSample / ForwardArgmax are synchronous Begin/End and bypass encode-ahead | metal/gumbel_sample.go:45-71; metal/model.go:2222-2250 | eval.cpp:29-69 | up to 3.6-4.2% (0.5B), ~2.8% (1.5B), ~0.9% (7B) [P] | add sample mode to execJob, A/B at T=1.0 |
| C-B02 | Minor, REVISIT | Residency verdict: MLX sets are requests under memory pressure and default to 0; goinfer's recorded bisect says pinning costs in proportion to set size; the premise was measured on paged MoE, not dense | resident.h:15-26; resident.cpp:221-237; allocator.cpp:100-105,254-262; metal/model.go:1581-1647; s6-alias-2026-09-24.md:197-198 | resident.cpp | 7B alias cost -1.4% [R]; recovery or regression unknown | per-encoder set on dense 7B, interleaved A/B n>=12; kill if no recovery of the 1.4% or +0.5 ms/token |
| C-B03 | Minor, REVISIT | MLX's spin fence (fast-synch) is not what the recorded shared-event test measured | fence.cpp:11-24,34-50; kernels/fence.metal; utils.h:220-222; metal/pagecost_sharedevent_test.go:47-64 | event.cpp is MTLSharedEvent | upper bound only: 7.8 ms/token = 5.4% of M26 at 30 boundaries [D]; no saving figure recorded | empty-CB round trip: commit+wait vs commit+spin, n>=1000; kill if saving < 25 us/boundary |
| C-N02 | Minor | Whole-prompt prefill is one command buffer: no cancel point, scratch O(M), watchdog behaviour unrecorded | metal/backend.go:763-813; metal/prefill.go:1116-1208 | device.cpp:604-622 (40-50 ops/CB) | not an effect finding | cross-ref area A |
| C-G01 | Minor (G) | TestMetal_CloseFreesMemory gates on RSS only, which UMA GPU buffers do not move | metal/close_leak_test.go; sibling tests 144-282 use CurrentAllocatedSize | none | n/a | make the RSS test also assert CurrentAllocatedSize |
| C-D01 | Minor (D) | Stale comments/specs: guardGIWFit "file-backed" comment, S6 "never swap" wording vs v15, attnFACoreCount=14 vs 16-core target, N-20 premise, ForwardArgmax contradictory comments, runtime-selftest spec unimplemented | listed in entry | none | n/a | read-only |
| C-N01 | Minor | Encoder.Dispatch binds through a fixed [16] scratch with no length check; widest call binds 15 | aikit metal.go:676-700,819-842; metal/batch.go:633-635 | none | n/a | add a len guard |
| C-N03 | Minor | PrefillLast allocates 26 fresh buffers per call from zero-filled Go slices (34.3 MB at 1.5B, M=512 [C]) | metal/prefill.go:1116-1208 | allocator.cpp BufferCache | recorded host overhead indistinguishable from zero (metal-prefill-decomp-2026-09-25.md:49-50); at most 0.5% [P] | not worth a probe unless short-prompt TTFT becomes a target |

---

### (c) Full entries

#### C-P01 [P, memory] Paged-expert f16 scale cache duplicates aliased scales (Major)

**Claim.** `buildGemma4MoELayer` and `buildMoELayer` precompute every expert's gate|up and down f16 scales into heap `[]uint16` slices (`gScaleCache`, `dScaleCache`) and keep them for the life of the model. On a v15 .giw those scales already exist in the mmap as WeightMat storage (decoder/serialize.go:88, giwVersion 15; `Int4F16()` returns them as views). The cache is a second copy, anonymous, therefore the only kind of memory in the load path that can swap.

**Evidence.**
- Cache build (at `844700f8`; all removed 2026-10-02): metal/gemma4_moe.go's paged branch (`int4DirectBytes(experts[ei])` into `gScaleCache[ei]`, same for down). `int4DirectBytes` (metal/model.go) goes through `decoder.Int4F32(w)` then `parallelF32ToF16` into a fresh `make([]uint16, ...)`. The bytes half already uses the aliased path (`int4DirectBytesOnly`, metal/model.go:575-581, via `w.Int4F16()`), so the scales are the only duplicated half.
- Consumption (at `844700f8`): the stage closure returns `gScaleCache[ei]` and the pread slot fill copies it with `copyU16sToBuf`, both in metal/gemma4_moe.go's paged branch; metal/moe.go's paged branch has the same cache for the generic paged MoE.
- The pread path's own comment said scales "stay f32 to f16 from the heap-resident q4s ... so they never fault the mmap" (metal/gemma4_moe.go at `844700f8`; corrected 2026-10-02). That is the v13 premise (f32 scales on heap); under v15 the heap f32 copy is gone and the cache is the only heap holder.
- Size, counted: ~186K scales per expert (the comment at metal/model.go:583-586 says "~186K scales" per expert) x 2 B x 30 layers x 128 experts = 185,856 x 2 x 30 x 128 = 1,427,374,080 B = 1.43 GB [C]. The Sep 12 audit N-20 recorded the f32 form of the same heap as "~2.85 GB on the 26B, 2x the bytes the GPU consumes" (audit-metal-2026-09-12.md:1384-1386), which is 1.43 GB at f16 [R]; the layer count and expert count are the M26 configuration, not re-read here (section (f)).
- Observed: the M26 untagged host heap is 1,948 MB in every arm, aliased or copied (s6-alias-2026-09-24.md:264) and S6 says it does not touch it (:269-270). The cache accounts for 1.43 of those 1,948 MB by count; the other ~0.5 GB is unattributed here.
- The fit guard does not price it: decoder/residentneed.go ResidentNeedBytes adds weights, the Metal host copy and KV, with mmap-aliased bytes subtracted (decoder/weightbytes.go:184-192); no term for a paged-expert scale cache.

**Why this is not already closed.** N-20 (audit-metal-2026-09-12.md:1384-1395) is marked FIXED because it removed the per-stage re-derivation, and its premise (an f32 heap copy to avoid re-converting) predates weights format v12/v15; s6-alias-2026-09-24.md:269-270 records the 1.95 GB heap as "the same in both arms, S6 does not touch it" and no later doc attributes it. I found no record that names the cache as retained anonymous memory.

**MLX analog.** None for the cache; MLX has no expert paging. The relevant MLX fact is that its weights are the allocation, with no second derived copy per expert.

**Delta and fix.** Stage scales from the aliased `Int4F16()` views (one fewer copy, but it faults the mmap on the stage path, the thing the pread path was written to avoid) or, preferably, pread the scale span with the nibbles into the slot (+12.5% pread bytes: scales are 2 B per 16 B group, i.e. 1/8 of the nibble bytes [C]) and delete both caches.

**Projection [P].** Footprint -1.43 GB at token 32 (48% of 2,967 MB) with no recorded basis for a smaller number: it is a count, subject to the 128-expert assumption. Decode: -6% to 0% on the regression side (the stage path moves 12.5% more bytes if pread is used; the staging share of the M26 token is not in my sources, so the band is wide). Swap effect: anonymous pages move to file-backed pages, which are reclaimable without swap.

**Bit-identity.** Identical: the same f16 bits come from the file instead of from a conversion of the same f16 widened to f32 and narrowed again (f32ToF16 of a value that began as f16 is exact).

**Probe.** On a v15 M26 .giw, aliased arm, token 32 as in s6-alias-2026-09-24.md: (1) heap profile or `vmmap` region table to attribute the 1,948 MB; (2) the same cell with the cache replaced by views/pread; footprint pairs n>=3 (footprint is stable), decode pairs n>=12 interleaved (recorded spread is about +-13% at n=4, s6-alias-2026-09-24.md:272). **Kill:** reduction under 1.0 GB (attribution wrong) or decode worse than -3% with the pread variant.

---

#### C-C01 [C, memory] Decoder fit guards ignore Metal's ceiling and KV precision (Major, conditional on a load the guard trims)

**Claim.** `guardGIWFit` (and `guardFit` for GGUF) size an unpinned context as the largest that fits free memory, up to the model's own window, priced at f32 KV unless `Options.KVPrecision == "f16"` (decoder/fitguard.go:366-423, 394-395, 742-757). Metal's real KV is f16 (metal/model.go:1306-1311, metal/backend.go:300-336) and its resident cap is a separate 32768 ceiling (metal/model.go:22-60). `Plan("metal")` has no ceiling term; only webgpu does (decoder/fitplan.go:233-243,275,306-307,322-328). The pinned value is passed as `opts.ResidentContext` (decoder/model.go:601-611) and `resolveMetalCtxCap` reads it as an explicit request (metal/model.go:44-60). Two consequences:

1. Inflation. A guard that trims on a tight machine hands Metal a pin of 12-30k positions. Without the trim the backend default is 4096 (decoder/fitplan.go:184). So the machine with less free memory allocates more KV. N-41 recorded the pins that did this: 12,109-29,666 positions on a machine with about 5 GB free, "3-7x over the ceiling" (audit-metal-2026-09-12.md:1595-1599). With the ceiling now 32768, none of those pins would be refused today; they would be allocated, which is the inflation case. KV bytes are linear in context, so the Metal KV at those pins is 3-7x the 4096-default KV [D].
2. Refusal. `resolveMetalCtxCap` returns an error above 32768 (metal/model.go:44-60). For a model whose window exceeds 32768, the guard's pin is the largest context that fits, which is above 32768 whenever the budget is large enough to hold 32768 f32-priced positions but not the whole window. Metal then refuses, BuildResident returns an error, and decoder/residency.go:1149-1168 prints one stderr line and continues on the CPU/staged path: the whole forward moves to CPU. A roomier machine thus gets CPU while a tighter one gets a large GPU KV. The refusal band for a model with per-position f32 KV cost k bytes and window W is budget in (32768 x k + 256 MiB, W x k + 256 MiB) (`prefillAttnScratchBudget` is 256 MiB, decoder/scratch.go:232) [C]. I did not read a config in the repo with W above 32768 and have no recorded run in that band.

**Why this is not already closed.** N-41's own text says "Fix (not attempted, out of scope for this pass)" (audit-metal-2026-09-12.md:1603-1607); the later cap raise (metalCtxCapDefault stays 4096, metalCtxCapMax 4096 to 32768, metal/model.go:20-35) moved the number the three failing tests tripped over but not the unawareness; `Plan("metal")` still has no ceiling (decoder/fitplan.go:233-243). `decoder/residency.go:1149-1168` makes the symptom a stderr line rather than a failure, so no test sees it.

**MLX analog.** MLX has no kernel context ceiling to reconcile; its limits are memory-driven (allocator.cpp:63-65 block and gc limits). Not a borrowable technique; the fix is local.

**Fix.** Give `Plan("metal")` and the two guards a backend ceiling (the webgpu ceiling's shape, decoder/fitplan.go:169), price Metal KV as f16 via `Model.ResidentKVBytes("metal", ...)` (which the Metal guard already uses, metal/backend.go:330-336) and make an auto-pin (unpinned flag, decoder/model.go:242-245) a ceiling that Metal clamps to rather than refuses.

**Projection.** Memory: for the inflation case KV rises from the 4096-default to the pin; bytes scale linearly [D]. Perf: the refusal case loses the whole GPU forward (not a percentage figure; I have no recorded CPU-versus-Metal decode ratio for these models and do not invent one).

**Bit-identity.** Not applicable to a guard change; the clamp changes capacity only.

**Status, 2026-10-01.** Fixed by T0.4 (§10, Track 0) on the Metal side: an auto-pin is now a ceiling Metal clamps to, and `Plan("metal")` has the ceiling. The guards' f32 pricing was kept on purpose; the Track 0 row says why and what it costs.

**Probe.** Unit test, no device: build a Config with MaxPositions 131072, inject `hostRAMAvailable` at several values, record `guardGIWFit`'s return, then call `resolveMetalCtxCap` with it. **Kill:** if no injected value yields a pin above 32768, the refusal half is not reachable (the inflation half stands on N-41's recorded pins). Companion probe for inflation: whether untouched KV pages count at allocation (section (f)), which decides whether the inflation costs footprint at load or only as the context fills.

---

#### C-B01 [B, P] On-device token feedback (async chain) (Minor on 1.5B/7B)

**Status, 2026-10-03: shipped on the branch.** Graded chain ÷ off **1.085×** on the 1.5B (9 of 9 pairs above 1); 0.5B
1.063×, 7B 1.010×. Record: `docs/tasks/task-metal-audit-2026-10.md`, "The night of 2026-10-02: results".

**What MLX does.** `async_eval` encodes graph n+1 on the host while graph n runs and commits with a completion handler; there is no host wait between graphs (eval.cpp:29-69). The commit policy is per-chip ops and MB per buffer (device.cpp:511-513, 604-622: 'p' 20/40, 'g' 40/40, 's' 50/50, 'd' 50/50, default 40/40; env MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER). That the next token's id is an array inside the graph is a property of mlx-lm, which is not in the snapshot, so I cite only the mechanism MLX provides (async commit without a wait), not the decode loop.

**What goinfer does.** `execLoop` (metal/model.go:2091-2156) commits token t, pre-encodes t+1's buffers while t runs, waits, copies 608 KB of logits, applies softcap, acks (a channel), and only then can the host argmax the logits and load the next embedding row. Greedy on Metal is that path: Metal has no `ResidentGreedy` (N-10 in the comment at metal/model.go:2216-2221; decoder/model.go:1929-1931 `hasGreedy` is false), recorded as host argmax of ~30 us on UMA. `ForwardArgmax` exists but is test/spec-only and synchronous (metal/model.go:2222-2250). The id therefore always round-trips through the host before the next token's first kernel can start, and that gap is exactly what encode-ahead leaves: 0.45 ms (1.5B), 0.50 ms (7B) [R] concurrency-mc3-s4-2026-09-27.md:35-36.

**Delta.** Chain the token on device for the greedy/device-picked modes: reuse the fused `gemv_w8a8_amax` + `argmax_finish` head (already used by `ForwardArgmax`), add a gather kernel that writes the next embedding row (with the arch embed scale, `loadEmbedRow`, G-02) from the device-resident table into the input buffer, and commit CB(t+1) behind CB(t) on the same queue (queue order, no event). The host then reads only the 4-byte token for stop checks. One speculative forward past an EOS or stop string is wasted per sequence; the KV slot it writes is overwritten later.

**Projection [P].** Removable part of the 0.45 ms: 0.30-0.45 ms/token. Basis: the non-GPU time left after pipelining is wake-up, logits copy, softcap, host argmax, embed load and commit, all of which sit between kernEnd(t) and kernStart(t+1); the lower bound assumes two thirds of it is removable. Percentages use token wall = GPU token time + 0.45/0.50 ms non-GPU (assumption: the decode-decomp figures are GPU-side): 0.30-0.45 / (12.90+0.45) = 2.2-3.4% on the 1.5B; 0.30-0.45 / (45.51+0.50) = 0.7-1.1% on the 7B; 0.30-0.45 / (5.22+0.45) = 5.3-7.9% on the 0.5B, which assumes the 1.5B's non-GPU time carries over (it is not recorded for the 0.5B). The 0.5B top of band would cross the 5% Major line; that is a projection on an unmeasured assumption, so the finding stays Minor.

**Park-line context.** The recorded MC3 batched-step executor was parked at about 4% (1.5B) and 1.3% (7B) (concurrency-mc3-s4-2026-09-27.md:38-40); this is the same bucket from the single-sequence side with a smaller scope, and a similar size on the 1.5B.

**Why not already closed.** I grepped docs and code for token feedback, on-device embedding gather and speculative commit and found only `pCopyVec` (metal/model.go:303, a batched copy), a qwen35 test comment about greedy self-feeding, and a host embedding-gather note (decoder/weightbytes.go:132); nothing records this as tried or rejected. N-27 (declined) concerns the logits copy alone, a ~30-60 us piece of this gap (audit-metal-2026-09-12.md:1466-1470).

**Bit-identity.** Identical: same kernels in the same order; only where the token id travels changes. The gather must reproduce `loadEmbedRow` exactly, including the arch scale and any learned position add (`addLearnedPos`). Tied-embedding and device-picked modes only; non-greedy modes that need full logits (penalties, processors, logprobs) keep the host path.

**Probe.** Read `kernStart(t+1) - kernEnd(t)` from the existing timestamps (`ReadTimes`, aikit metal.go:808-814) across 200 tokens on 1.5B and 0.5B to get the real GPU-idle gap; that number replaces the band. **Kill:** median gap under 0.15 ms.

**Status, 2026-10-02 (branch `metal-audit`): built, bit-identical, pending its night grade.** T1.3 measured the gap at
0.583 ms (0.5B) and 0.663 ms (1.5B), so the finding stood.

The build is `metal/greedy_chain.go`:
- The embedding is gathered on the device (`embed_gather_i8`), from the LM-head table for a tied head. For an untied
  one it comes from a device copy of the int8 embedding table, made on first use and priced against the memory guard.
  The bench Qwen GGUFs are all untied.
- No embedding scale or learned position: the chain declines those.
- The fused argmax head ends each buffer, and token t+1's buffer is committed before the host waits for t. On the identity
tests, 0 tokens and 0 K/V elements differ across 80 tokens, and 150 tokens through `Generate`, a reused prefix
included, are all served by the chain. The grade and its rule are in the task doc ("C-B01: built, pending its grade").

---

#### C-P02 [P] Device-sampled decode bypasses encode-ahead (Minor)

**Status, 2026-10-03: shipped on the branch.** The sampled chain at T = 1: **1.066×** on the 1.5B (9/9), 1.160× on
the 0.5B (above the projected 3.6–4.2% because the host round trip went too), 1.022× on the 7B.

**Status, 2026-10-02 (branch `metal-audit`): built as the sampled chain, bit-identical, pending its night grade.**
Rather than adding a sample mode to `execJob`, the C-B01 chain was given `ForwardSample`'s draw: each buffer ends with
the full LM-head row and the two gumbel dispatches, and the next one is committed before the host waits. That closes
the missing encode-ahead and the host round trip together. The tests, the −1 recovery and the grade's rule are in the
task doc ("C-P02: built, pending its grade").

**Claim.** `ForwardSample` (metal/gumbel_sample.go:45-71) and `ForwardArgmax` (metal/model.go:2222-2250) each do `Begin ... End` on their own encoder, synchronously, and never enter `execLoop`. The decode loop routes temperature-only sampling above 0.2 through `ForwardSample` (decoder/model.go:1989-1993, 2157-2174); T at or below 0.2 goes to the optimistic-forward overlap (decoder/spec_optfwd.go:30-31, optFwdMaxTemp 0.2); top-k/top-p have no Metal device path (no `TopKAvailable`/`ForwardTopK` in metal/*.go) and use the pipelined full-logits `Forward`. So at the serve default temperature 1.0 [R] audit-metal-2026-09-12.md:1487-1489 (N-28), the shipped device sampler gets the synchronous path while greedy gets the pipelined one.

**Recorded numbers.** R7b device draw ran at 0.958-0.964x of greedy on the 0.5B, host draw at 0.799-0.842x (r7b-metal-mac-2026-09-20.md:215-216, 263). Greedy there has encode-ahead; the device sample does not, so a part of the 3.6-4.2% shortfall to greedy is the missing encode-ahead.

**Projection [P].** Up to 3.6-4.2% on the 0.5B (the whole recorded gap, an upper bound because the gumbel dispatches themselves also cost something), about 2.8% on the 1.5B and 0.9% on the 7B (scaled by the ratio of the 0.5B token, 5.22 ms, to 12.90 and 45.51 ms [D]).

**Delta.** Add a sample mode to `execJob` that appends the two gumbel dispatches (and passes seed and draw through the job) so the t+1 buffer can be pre-encoded; the per-call uniforms (`uGumbelInvT`, `K0/K1`, `D0/D1`) change per token, so they must be written after WaitDone(t), as `uPos` already is.

**Bit-identity.** Identical: same dispatches, same seed and draw.

**Probe.** A/B at T=1.0 on the 0.5B and 1.5B, n>=12 interleaved. **Kill:** gain under 1%. Cross-reference: area E owns sampling; this entry is the executor half.

---

#### C-B02 [B] Residency sets and the wired limit: verdict and a REVISIT (Minor)

**Direct answers.**

*Does MLX keep weights resident with a residency set?* Only if asked. `set_wired_limit` defaults to 0, "i.e. nothing is wired unless asked for" (resident.h:15-16); the manager spreads allocations over several size-capped sets, each with a standing `requestResidency()`, attached to every command queue (resident.h:18-23; resident.cpp:221-237). It is a request, subject to GPU memory pressure: the cap per set exists because a set can lose residency under GPU memory pressure and only that set must be re-made resident (resident.h:18-22).

*Does goinfer?* Only for the paged-MoE slot pool (metal/model.go:1581-1647, scope "slots"), attached per encoder on phase-2 command buffers (`Encoder.UseResidencySet`, aikit residencyset.go:125; metal/moe.go:1016, metal/gemma4_moe.go:594), not on the queue. Dense weights are not in any set; the NoCopy-aliased pages are file-backed and "wired only while a command buffer uses them" (metal/alias.go:31-35).

*Effect on the 16 GB machine under memory pressure.* The swap incidents in the never-swap record were anonymous pages (host copies of weights); S6's alias removed them (1.5B 90 vs 1,013 MB, 7B 105 vs 4,134 MB at token 32 [R] s6-alias-2026-09-24.md:188-189). A residency set over aliased weights would hold reclaimable page-cache pages wired continuously, which makes them non-reclaimable for the rest of the system, the wrong direction under pressure. The recorded memory-hog arm (7B, a 6.1 GiB random hog): alias 0 MB swap versus +56 MB for the copy, decode 21.73 vs 21.81 tok/s (s6-alias-2026-09-24.md:147-159); the arm says the alias already behaves, so a set has no swap benefit to add.

*How does MLX's wired limit work and could goinfer set the equivalent?* `set_wired_limit(limit)` (allocator.cpp:254-262) rejects values above `max_recommended_working_set_size` and calls `MetalAllocator::set_wired_limit` (allocator.cpp:100-105), which resizes the residency-set manager's budget. It is not the `iogpu.wired_limit_mb` sysctl. Goinfer has every binding needed: `NewResidencySet`, `Add`, `Commit`, `RequestResidency`, `Queue.AddResidencySet`, `Encoder.UseResidencySet` (aikit residencyset.go:24-29,98-125) and no wired-limit code of its own (grep over metal/ and decoder/ found none).

*The cost goinfer measured.* Queue-level attach of a ~3 GB set cost +2.07 ms per command buffer, +62 ms/token on paged phase 1 (metal/model.go:1641-1643); pinning more than the slot pool regressed phase 1 in proportion to the pinned size, read/write-agnostic (metal/model.go:1584-1591, 1450); pinned slots took phase-2 idle from 9 to 0.44 ms/CB. That is about 0.7 ms per CB per GB [D]. MLX attaches to every queue but runs 7-8 command buffers per token at goinfer's op count (below); goinfer's dense decode runs 1, but a paged-MoE token submits per layer, which is why the per-encoder attach was needed (M-14 closed).

**REVISIT (stale premise).** The bisect was measured on the paged-MoE path (many small command buffers per token, pread-dirtied slot buffers) on an otherwise loaded 16 GB machine; it was never run on dense decode with NoCopy-aliased weights. The only recorded dense symptom of the alias is the 7B's -1.40/-1.49/-1.19% decode cost, which grew from about -0.3% when the LM head and scales were also aliased (s6-alias-2026-09-24.md:197-198). If that cost is per-command-buffer wiring of aliased pages, a per-encoder set over the aliased weights could remove it; if the cost is set-size overhead as the bisect says, the same set would add about 0.7 ms per GB per token (about 5% of a 1.5B token for 0.9 GB [P], band 0 to +5%, basis: dense extrapolation of the paged figure, unmeasured).

**Cheapest probe.** Dense 7B decode, per-encoder set containing the aliased weight buffers (attached the same way as phase 2), interleaved A/B n>=12, depth 128, with `vm_stat` pages-free logged. **Kill:** no recovery of the 1.4% (about 0.64 ms/token [D] on the 7B's 45.51 ms), or a regression of 0.5 ms/token or more.

**Bit-identity.** No numeric effect.

---

#### C-B03 [B] MLX's spin fence versus the recorded shared-event premise (Minor, REVISIT)

**What MLX has.** Two sync primitives. `event.cpp` is the `MTLSharedEvent` wait/signal path (same mechanism as the recorded test). `fence.cpp` is a separate fast-synch fence: the GPU kernel writes a word and the CPU spins on it (fence.cpp:11-24, 34-50; kernels/fence.metal; enabled only by `MLX_METAL_FAST_SYNCH`, utils.h:220-222, default 0; macOS 15 and Metal 3; needs MSL 3.2 `coherent(system)` memory).

**What was recorded.** `TestPageCost_sharedEventReal` measured the MTLSharedEvent handshake at ~0.26 ms/boundary against ~0.23 ms for per-layer submit on the dense 1.5B, "recovers ~0%" (metal/pagecost_sharedevent_test.go:47-64). G-05 (audit-metal-2026-09-12.md:1154-1160) flagged that shape as lacking the cost it targets, and the Sep 12 closing list says M-11 and its shared-event re-run on the paged shape are "unchanged, still fully open" (:1760-1766). No re-run exists.

**Premises that moved.** (1) The recorded boundary cost on the paged 26B was ~14-15 ms (M-11); the slot residency set cut phase-2 idle from 9 to 0.44 ms/CB (metal/model.go:1584,1626). The target cost the event design was meant to remove has therefore changed by an order of magnitude. (2) The tested primitive was the event, not the spin fence; a fence changes the wake path (user-space spin versus kernel-mediated `waitUntilCompleted`), which is a different quantity from the event-versus-commit comparison the test made.

**Bound, not a projection.** If a fence made a boundary free: 30 boundaries (one per layer) x 0.26 ms = 7.8 ms/token, 5.4% of the M26 token (mean of the four aliased arms in s6-alias-2026-09-24.md, 6.80/6.56/7.17/7.09 tok/s, i.e. 144.8 ms) [D]; with M-11's 61-81 command-buffer boundaries per token the bound is 14-19 ms. Neither is a saving figure: the saving per boundary is not recorded and the paged path has a real host dependency at each boundary (the host reads the router's choice and preads the experts), so a fence shortens the wake but cannot remove the host step.

**Dense decode.** The same fence could replace the one `waitUntilCompleted` wake per token inside C-B01's 0.45 ms gap; C-B01 removes that wake entirely for greedy, so the fence matters there only where the host must read before continuing (stop checks, paged phases).

**Probe.** aikit microbench in the style of pagecost_sharedevent_test.go: empty command buffer commit+`waitUntilCompleted` versus commit+spin on a coherent word written by a one-thread kernel, n>=1000 each, report p50/p99. **Kill:** saving under 25 us per boundary (under 10% of the recorded 0.23-0.26 ms). Availability caveat: MSL 3.2 is not confirmed for the aikit library compile path (section (f)).

**Bit-identity.** No numeric effect.

---

#### C-N02 [N, cross-ref A] Whole-prompt prefill is a single command buffer

`PrefillLast` (metal/backend.go:763-813; metal/prefill.go) encodes the whole padded prompt into one command buffer and waits once. MLX splits work into 40-50 ops per buffer (device.cpp:604-622) and can abort between buffers. Here there is no cancel point between the first and last layer of an M-token prompt, per-request scratch is O(M) (metal/prefill.go:1116-1208), and the GPU watchdog behaviour for a long single buffer has no record (section (f)). Chunked prefill exists only under MC3 (area E). Area A owns the effect; I note only the runtime-structure fact. Bit-identity of a split: identical if the split falls between dispatches (no kernel changes).

#### C-G01 [G] RSS-only free test

`TestMetal_CloseFreesMemory` (metal/close_leak_test.go) asserts on RSS. On UMA a GPU buffer is not necessarily in RSS, so a leak of device buffers does not move the number the test reads; sibling tests (metal/close_leak_test.go:162-300) read `CurrentAllocatedSize` and so can fail. Sep 12 G-06 closed two of its gates; this is the residue. Fix: assert `CurrentAllocatedSize` returns to its pre-load value (or below a stated slack) in the same test.

**Status, 2026-10-01.** Done: the test reads `CurrentAllocatedSize` through its own probe Device and fails if it ends more than 16 MiB above the post-warm-up value. Measured on the 0.5B: 655,360 bytes before and after every cycle, 401 MB with a resident built. With `Close` made to skip `ReleaseAll`, it fails at +1.6 GB (4 × 401 MB); RSS also failed, but saw only +631 MB and +1022 MB of it in two runs.

#### C-D01 [D] Stale comments and specs

1. decoder/model.go (S4 comment before `guardGIWFit`, :571-579) says a .giw's weights are file-backed so no fit check is made at all; since S6 and v15 that is true of the weights but not of the paged scale cache (C-P01) or the ~1.95 GB heap.
2. The never-swap and S6 wording that the load path "never swaps" holds for aliased dense weights (0 MB swap, s6-alias-2026-09-24.md:147-159) but not for the M26's anonymous heap; the M26 hog arm was deliberately not run (:148).
3. metal/model.go:2698-2703 `attnFACoreCount = 14` against the 16-core M1 Pro target (area B owns the effect; the constant is a runtime-visible stale value).
4. N-20's premise (an f32 heap copy of the scales exists) no longer holds under v15 (C-P01).
5. `ForwardArgmax` carries two contradictory comments: "NOT actually production's greedy decode path ... exercised only by tests/gates" (metal/model.go:2216-2221) and "a genuine production entry point (the fast-greedy path)" (:2051-2053). The first is right (no `ResidentGreedy`, decoder/model.go:1929).
6. docs/tasks/task-metal-runtime-selftest.md specifies a runtime self-test that is not implemented. (2026-10-01: folded into docs/tasks/task-hardware-coverage-2026-10.md §H2; the original is archived in docs/completed/.)

#### C-N01 [N] Fixed-size binding scratch

`Encoder.Dispatch` and `DispatchTG` fill `idScr [16]` and `offScr [16]` and call one batched `setBuffers:offsets:withRange:` (aikit metal.go:676-700,819-842) with no length check. The widest dispatch in goinfer binds 15 buffers (metal/batch.go:633-635). One more binding on that kernel overruns the array. Add a bounds check or panic with the kernel name at encode time.

#### C-N03 [N] Per-request prefill scratch

`PrefillLast` allocates 26 buffers per call (dense; 33 with MoE): six f16 activation buffers from `make([]uint16, ...)` (zero-filled Go slices copied into a fresh MTLBuffer by `NewBufferU16s`), `posB`, `moeDst` and eighteen one-word uniforms, released at return (metal/prefill.go:1116-1208). Counted at the 1.5B (H=1536, I=8960, qkvDim 2048, qDim 1536), Mpad=512: xF 1,572,864 + normF 1,572,864 + qkvF 2,097,152 + ctxF 1,572,864 + guF 18,350,080 + dqF 9,175,040 = 34,340,864 B = 34.3 MB [C], i.e. 2,096 16 KB pages touched twice (Go heap, then MTLBuffer). MLX's `BufferCache` and 1 MiB small-buffer heap (allocator.h:51-52, allocator.cpp) exist to avoid this. Effect bound: the recorded prefill replica found host-side work "(f16 conversion, buffer allocation, encoding) indistinguishable from zero" against a ~1,580 ms K=512 wall (metal-prefill-decomp-2026-09-25.md:49-50, 94-95), so at most 0.5% [P] (basis: that statement plus the 17% wall spread). Not pursued; a pool is bit-identical.

---

#### Direct answers: launch overhead per token, goinfer versus MLX

**Count.** Dense Qwen2, 28 layers: 11 dispatches per layer [C] (attention half 7: norm, qkv GEMV with bias, rope, kv store, attention, o-proj with residual, plus the f32-to-f16 conversion on the f16 lane; FFN half 4: norm, fused gate|up GEMV, SwiGLU, down with residual; metal/model.go:2920-3072 and 2357-2445), plus 2 at the head: 11 x 28 + 2 = 310 dispatches/token (338 when `attention_fa` adds partial+combine, i.e. +1 dispatch x 28 layers at depth >= 1536: 310 + 28 = 338). `Encoder.Dispatch` makes three purego calls (setComputePipelineState on change, one batched setBuffers, dispatchThreads; aikit metal.go:819-842); every adjacent pair of dispatches has different pipelines, so the pipeline set is not skipped; 4 of 11 per layer are `DispatchTG` with one more call (setThreadgroupMemoryLength): 11 x 3 + 4 = 37 per layer. 37 x 28 + 2 x 3 = 1,042 transitions per token, plus about 10 command-buffer-level calls (create, encoder, end, commit, wait, status, and four timestamp reads in `ReadTimes`, N-31). The Sep 12 audit counted "~1,030" (audit-metal-2026-09-12.md:1612). [C]

**Cost per transition.** Recorded host encode ~0.9 ms (metal/model.go:450-452) / 1,042 = 0.86 us each [D]. The aikit comment records that the two trims already shipped removed only ~0.5 ms of a ~2.4 ms/token overhead and that "the rest is GPU-side per-Dispatch latency, not Go-side msgSend" (aikit metal.go:686-689). Today's GPU floor with every pipeline no-op'd is 0.86 ms on the 1.5B = 2.8 us/dispatch [D]. After encode-ahead the visible host side is 0.45 ms [R].

**MLX.** MLX encodes the same kind of per-op calls from C++ with no purego layer (per-call cost not measured here), puts 20-50 ops in a command buffer by chip (device.cpp:604-622) and commits asynchronously (eval.cpp:29-69). At goinfer's 310 dispatches that is 7 command buffers at 50 ops and 8 at 40 ([D]; MLX counts primitives, not dispatches, so the true count differs); which bucket an M1 Pro falls in is not statically known (section (f)). MLX does not wait between graphs; goinfer waits once per token.

**setBytes batching.** Each `setBytes:length:atIndex:` is one more msgSend, and goinfer's per-token uniforms (position, key count) are 1-word MTLBuffers written by plain Go stores into unified memory, which cost zero transitions today. Moving them to setBytes would add up to one transition per dispatch that carries one and remove none: they already ride the single batched setBuffers. Not a reduction. Bit-identity: identical.

**Argument buffers.** One setBuffer(s) call per dispatch becomes one setBuffer(argument-buffer, offset) call: the transition count is unchanged. They also drop Metal's implicit residency of the referenced buffers, so each would need `useResource` or a residency set (C-B02's cost). Not a reduction.

**Fewer encoders.** Already one encoder per command buffer (aikit metal.go:676-700). Nothing to merge.

**What would reduce transitions.** Only an indirect command buffer (one `executeCommandsInBuffer` per token): a recorded negative (red-october.md:466; Sep 12 §5 :1614-1615). Premise check in the next block.

**Ceiling.** Host side visible after encode-ahead: 0.45 ms (3.4% of a 1.5B token wall [D]). GPU-side floor: 0.86 ms (6.7% of 12.90 ms; 2.3% of the 7B's 45.51 ms; 12.4% of the 0.5B's 5.22 ms [D]), unaffected by any host-side batching. Removing 10% of dispatches (31 of 310) would save at most 31 x 2.8 us = 0.086 ms = 0.67% of a 1.5B token [P], band 0-0.7%, basis: floor divided by count.

#### Direct answers: concurrent dispatch and barriers

**What MLX does.** `DispatchTypeConcurrent` encoder (device.cpp:580), buffers allocated `HazardTrackingModeUntracked` + `StorageModeShared` (allocator.cpp:15-16), and a memory barrier is inserted only when an op's inputs overlap a previous op's outputs (read-after-write, write-after-read; `maybeInsertBarrier`, device.cpp ~393-410). Independent ops overlap on the GPU.

**What goinfer has.** One serial encoder; every buffer hazard-tracked. M-16 tested untracked buffers on the serial encoder and found 99.6-101.5% of baseline (aikit hazard_tracking_probe_test.go; audit M-16 closed NEGATIVE, :1779), as expected: a serial encoder already orders every dispatch.

**Independent branches in a dense layer: none.** The 11 dispatches per layer form one read-after-write chain: norm (writes the normed activation) feeds the qkv GEMV (writes `r.qkv`), rope2 rotates Q and K in place in `r.qkv`, kv_store reads K/V from `r.qkv` (one kernel for both K and V), attention reads `r.qkv` and the cache kv_store just wrote, the o-proj consumes the context, then the FFN chain norm to gate|up to SwiGLU to down+residual. q/k/v are already one fused GEMV and gate/up one fused GEMV (metal/model.go:2920-3072, 2410-2445), so the pairs a concurrent encoder would overlap have already been merged. The remaining theoretical pair is kv_store's V half against rope2, merged in one kernel. With a barrier at every dispatch a concurrent encoder gains about 0 on this path [P], band 0 to +0.3%, basis: the M-16 result on the serial encoder plus a chain with no unbarriered pair. The one family with independent branches is Gemma-4's parallel dense||MoE FFN (metal/model.go:2549-2640 comment); that is area D's.

**Bit-identity.** Concurrent dispatch of independent kernels is bit-identical (no reduction is reordered inside a kernel).

#### Recorded negatives: do the premises hold on today's kernels?

| Negative | Recorded premise | Status today | Evidence |
|---|---|---|---|
| ICB (red-october.md:466; Sep 12 §5 :1614-1615) | Per-dispatch cost is GPU-side latency, not Go-side msgSend; per-token values (pos, key count, attention plan) vary | HOLDS. Recomputed bound: host encode 0.9 ms hidden to 0.45 ms visible, so an ICB can save at most that 0.45 ms (3.4%, 1.5B); the 0.86 ms floor is GPU-side and was measured with the encoder path unchanged. The attention plan switches at depth 1536 (11 vs 12 dispatches per layer), so one ICB per plan would be needed | metal/model.go:450-452; decomp doc :43; model.go attnFACoreCount region 2495-2507 |
| Unretained references (MLX uses them, device.cpp:323,559) | Retain/release per binding is small against the risk of a freed buffer | HOLDS. Binding count per token is unchanged (310 dispatches, same as Sep 12); aikit binds with `commandBuffer` (retained refs). Lifetime would be safe (the ledger frees at Close, prefill scratch is released after a waiting `End`), so this is a measured-small-gain close, not a safety close; I did not find a gain figure that would reopen it | aikit metal.go:819-842; metal/prefill.go:1149-1208 |
| Megakernel / dispatch-count fusion | A chain of 11 with reduction-order sensitivity | HOLDS. qkv and gate|up are already fused; ceiling for any further count reduction is the 0.86 ms floor (0.67% per 10% of dispatches [P]); fusing across a reduction boundary changes bits | red-october.md:466; decomp doc :43 |
| Shared event (metal/pagecost_sharedevent_test.go:47-64) | Event handshake ~0.26 vs per-layer submit ~0.23 ms/boundary, measured on a dense 1.5B | STALE in shape. Target cost on the paged shape fell from ~9-15 ms to 0.44 ms/CB idle after the slot residency set; the primitive tested was the event, not MLX's fence; re-run never done (M-11 open). REVISIT as C-B03 with the cheapest probe above | audit :1154-1160, :1760-1766; metal/model.go:1584,1626 |
| M-16 untracked hazard tracking | 99.6-101.5% of baseline | HOLDS, limited to the serial encoder; says nothing about a concurrent encoder (which has no unbarriered pair on the dense chain, above) | aikit hazard_tracking_probe_test.go |

---

### (d) Carry-forward (Sep 12 IDs in this area)

| ID | Sep 12 claim | Status | Evidence |
|---|---|---|---|
| M-01 | `ResidentPrefillKV` missing on Metal: every sequential prompt token ran the full LM head and 608 KB readback | CLOSED-VERIFIED | `ForwardNoLogits` implemented, pipelined with a `noHead` bit on `execJob`; paged-MoE falls back to the head-bearing path by design (metal/backend.go:577-603; metal/model.go:513,1963,2006-2038) |
| M-07 | Three host copies of every dense int4 projection on a GGUF/safetensors load | CLOSED-VERIFIED for the row4-skip half; host-release half declined as permanent | decoder/weights.go:720-724,769-775,805-808,1208-1213,1522-1526; audit :1753-1756, :396, :432-442 |
| M-11 | Paged decode pays a ~14 ms command-buffer boundary 61-81x per token | OPEN. The boundary cost fell (9 to 0.44 ms/CB idle, M-14), shared-event re-run not done | audit :1766; metal/model.go:1584,1626 (area D owns it) |
| M-12 | Serial per-expert staging | CLOSED, cross-expert half only (as recorded) | audit :1763-1766 |
| M-13 | Pager engages only with explicit slots | CLOSED (auto-sized slots) | audit :1760-1762 |
| M-14 | Residency set rides every command buffer (+62 ms/token) | CLOSED, wiring verified live; saving not re-measured | metal/model.go:1641-1647; metal/moe.go:1016; metal/gemma4_moe.go:594; aikit residencyset.go:117-125 (TestEncoder_useResidencySet) |
| M-16 | Every buffer hazard-tracked, every encoder serial | NEGATIVE-CLOSED (serial-encoder premise) | aikit hazard_tracking_probe_test.go; audit :1779 |
| C-05 | `Run1DBatchTG`/`Run1DTG` own an autorelease pool without the OS-thread pin | CLOSED-VERIFIED | aikit metal.go:954-955 and 988-992 (`runtime.LockOSThread`, "Missed here" note) |
| G-05 | Shared-event verdict measured on a shape without the cost | OPEN (no re-run); see C-B03 | audit :1154-1160 |
| G-06 | Device-ledger free assertions pass by construction | CLOSED for two gates; residue C-G01 | metal/close_leak_test.go:162-300 |
| G-10 | C-09 status latch inert on Apple silicon | pointer closed (host pre-checks are the gate; `recordExecErr` sites at metal/gumbel_sample.go:69-71, metal/model.go:2253) | audit :1244 |
| N-20 | f32 heap scales re-derived per stage | CLOSED, but its replacement is C-P01 and its premise is stale under v15 | metal/model.go:575-581; metal/gemma4_moe.go:291-295 |
| N-21 | Expert slots zero-initialised | FIXED (`NewBufferLenOf`) | audit :1407-1412 |
| N-27 | 608 KB logits memcpy on the pipe path | DECLINED, reason still valid (decoder/spec_optfwd.go:202-211) | audit :1466-1480 |
| N-28 | No CPU-sampler overlap above T=0.2 | PARTLY ADDRESSED by R7b device sampler; executor half remains (C-P02) | r7b-metal-mac-2026-09-20.md:215-216 |
| N-29 | Load path rebuilt words byte by byte | FIXED (bulk copy in `bytesToU32`) | metal/model.go:558-565 |
| N-31 | Four GPU timestamp reads per production token | PRESENT (counted in the 10 command-buffer-level calls) | aikit metal.go:808-814 |
| N-34 | metal_copy / upload_batch unused and unfenced | UNCHANGED, still correct on UMA | aikit gpu/metal_copy.go, metal_upload_batch.go |
| N-36 | KV charged for DeltaNet layers | FIXED | decoder/residentneed.go; memory-accounting-metal-2026-09-25.md:38-46 |
| N-41 | Auto-pin ignores Metal's ceiling | PARTIAL, live in a new form | C-C01 |

M-08 and C-04 belong to area E and are not carried here.

---

### (e) Checked and found correct

- Decode structure: one command buffer, one commit, one serial encoder, one wait per token; 310 dispatches and 1,042 transitions counted against the Sep 12 "~1,030".
- `Encoder.Dispatch`: pipeline set only on change, one batched `setBuffers:offsets:withRange:`, uniforms as 1-word buffers; retained-reference command buffers (aikit metal.go:819-842). Only the missing length check (C-N01).
- `execLoop` (metal/model.go:2091-2156): OS-thread pinned, one long-lived autorelease pool drained every 64 tokens, `attnPlan` mismatch drops the pre-encoded buffer, `SetAdapter` tears the executor down because the t+1 buffer bakes in LoRA state (C-07), paged MoE bypasses the executor (:1750-1758).
- NoCopy alias: MAP_SHARED read-only mapping, `VM_INHERIT_NONE` (decoder/giwmap_darwin.go, forkinherit_darwin.go); alias.go reasons correctly about COW wiring (:28-36) and the gate requires fused groups, singles, f16 scales and the int8 head each aliased with 0 heap-backed/unaligned/non-adjacent (s6-alias-2026-09-24.md:211-212, 258). Footprints at token 32, aliased vs copied: 1.5B 90 vs 1,013 MB, 7B 105 vs 4,134 MB, M26 2,967 vs 4,485 MB [R] (:188-189, :262).
- Fit accounting: `ResidentHostCopyBytes` subtracts mmap-aliased bytes (decoder/weightbytes.go:184-192), so the guard does not over-price an aliased .giw (recorded 1.40 GB for the v14 1.5B). The Metal guard is min(70% of hw.memsize, live available) (metal/backend.go:69-117). `TestResidentKVBytes_matchesMetalAllocation` 5/5 exact (memory-accounting-metal-2026-09-25.md:38-41).
- Residency: scope "slots" per-encoder attach, verified live by aikit's test; the bisect rationale in metal/model.go:1584-1647 is internally consistent with the recorded figures.
- `ForwardSample`: seed/draw come from the sampler's own stream (`NextDraw`), the infinite-`invT` case degrades to host argmax exactly (metal/gumbel_sample.go:45-71).
- Optimistic-forward: the logits copy before the overlap (decoder/spec_optfwd.go:202-211) and the 0.2 cap are consistent with the recorded CUDA race and the T ladder.
- aikit `metal_copy.go`, `metal_upload_batch.go`: host memcpy, unused by goinfer, correct on UMA.
- MLX comparison facts used: device.cpp:323,559 (unretained), :580 (concurrent), :511-513 and :604-622 (commit policy), allocator.cpp:15-16 (untracked + shared), resident.h:12-26, resident.cpp:221-237, fence.cpp:11-50, event.cpp, eval.cpp:29-69.

---

### (f) Not settleable statically

1. The GPU-idle share of the 0.45 ms non-GPU time (C-B01): needs `kernStart(t+1)-kernEnd(t)` on a device.
2. Whether untouched KV pages count in the footprint at allocation (decides C-C01's inflation cost at load versus as the context fills).
3. Whether the M26 scale pages evict under nibble-pread pressure, and the real attribution of the 1,948 MB heap beyond the 1.43 GB cache (C-P01 probe).
4. The 128-expert and 30-layer M26 configuration used in C-P01's count (taken from the record's "N=128 cliff", not from a config file in the snapshot).
5. The mechanism of the 7B NoCopy -1.4% decode cost: per-command-buffer wiring, TLB reach, or something else (C-B02).
6. GPU watchdog behaviour for a whole-prompt single command buffer on long prompts (C-N02).
7. The M26 hog arm (deliberately not run, s6-alias-2026-09-24.md:148).
8. The N-41 refusal band end to end: needs `hostRAMAvailable` injection or a real model with a window above 32768; `cmd/` (the `-ctx` default and where `Options.ResidentContext` is set from flags) is not in the snapshot.
9. MLX's per-chip commit bucket for an M1 Pro (device.cpp:604-622 keys on a runtime architecture string) and whether MLX binds small constants with setBytes (the compute-encoder wrappers were not in my read set).
10. Whether MSL 3.2 `coherent(system)` is available through aikit's library compile path (C-B03), and the macOS floor for the target machine.
11. mlx-lm is not in the snapshot, so MLX's token-feedback loop and any `set_wired_limit` call it makes were not checked; I cite only the MLX mechanism.


## 7. Area D — MoE, paging, DeltaNet, gpt-oss, Gemma-4 MoE, routing, MXFP4

Reviewer D. goinfer HEAD `844700f8` , aikit v1.51.0, MLX HEAD `9c3d355`
(`mlx/backend/metal/`). Static review only: nothing was built or timed. Number provenance
is marked (rec) recorded with doc:line, (cnt) counted from code shape with the count shown, (mlx)
MLX source file:line, (proj) projection with band and basis. Memory sizes recorded as "MB" in the
repo for M26 experts (3.19) equal 3.19 MiB by my count (5.95M weights x 4.5 bits); I keep the repo's figure.

Geometry used for counts (rec): M26 = Gemma-4-26B-A4B, 30 layers all MoE, hidden 2816, 128 experts,
moe_inter 704 (`docs/measurements/prefill-moe-m26-2026-09-04.md:20`), top_k 8 (N=8 is "top_k itself",
`docs/tasks/red-october.md:424`), 3.19 MB per expert (`metal/gemma4_26b_paged_test.go:128,168`).

### (a) Summary

1. The largest gap in my area is paged-MoE prefill. A paged MoE prompt runs as M paged decode tokens, each
   with 2L+1 = 61 synchronous command buffers on M26. Recorded TTFT is 16.8-18.7 s for a 96-token prompt
   (5.1-5.7 prompt tok/s, about the decode rate), so a 512-token prompt is about 90-100 s by linear
   extrapolation. Expert-major through the pool is unbuilt and has three more blockers per family (D-P01).
2. Qwen3.5/3.6/3.8 hybrids have no batched prefill at all (`metal/model.go:1027-1029`, `r.dnet == nil`). MLX has a
   T-loop recurrence kernel with one state read and one state write per call, plus a C=8 chunked kernel
   for T>8. The dense batched/sequential ratio recorded on this machine is about 10x (D-B01).
3. MoE expert-major prefill is default ON on a gate taken on a dense 1.5B, and its own tests are a tiny
   Mixtral fixture at cosine >= 0.95 with no mutation check. It is not bit-identical in three counted
   ways, one of which (the router) is a discrete choice (D-G01).
4. The host-grouped expert loop issues 5 dispatches per active expert and syncs once per layer. MLX groups on
   the device (sorted gather, offsets, tile scheduler, bm=16 tiles). Bit-identical if the R16 inner loop is
   reused (D-B02).
5. `moe_route` / `route_gptoss` run on one GPU thread. The Sep 12 record puts that at about 10% of a
   fitting ~5 ms MoE token and calls it deliberate for a reason that does not need one thread (D-P04).
6. gpt-oss MXFP4 is decoded to f32 and requantized to int4 at load. 0.6 GB counted over the 20B's experts,
   plus a fidelity cost that is not recorded (D-B03).
7. M-11's "~14 ms per command buffer" premise cannot hold on the M26 paged token: 61 x 14 ms = 854 ms
   against a recorded 128-167 ms/token. The paged path after S3-S6 is stable at N=8 (swap flat, 6.0-7.8
   tok/s) but N>8 and M35 on Metal are unrun (D-P02).
8. Stale text in 10 places, notably red-october §2.6 and R11(c), `benchmarks.md:1976-1985, 2262`, and
   `metal/gptoss_kernels_test.go:8-9` (D-D01).

### (b) Findings table (ordered by effect)

| ID | Sev | Claim | Evidence | MLX analog | Band | Probe |
|---|---|---|---|---|---|---|
| D-P01 | Major | Paged MoE prefill = M paged decode tokens; no expert-major through the pool | `metal/model.go:1027-1029`; `m26-alias-fork-collapse-2026-09-24.md:177-180`; `metal/prefill.go:1313-1469` (non-paged only) | gather_qmm_rhs grouping only; MLX has no pager | 3-15x TTFT at M=512 (proj) | `moe_prefill_measure_test.go` on paged M26 at M=64/128/512 |
| D-B01 | Major | No batched DeltaNet prefill; hybrids prefill as M decode tokens | `metal/model.go:1027-1029`; `benchmarks.md:1913-1926`; `metal/deltanet.go:129-149` | `gated_delta_update.h:271-366` seq, `:169-265` chunk | 3-8x TTFT on resident hybrids (proj) | T-loop recurrence kernel + batched conv/gates on `qwen35_resident_parity` fixture |
| D-G01 | Major | Expert-major prefill ships default ON on a dense gate; MoE tests cannot see its failure modes | `metal/prefill.go:1314`; `metal/backend.go:611-658`; `metal/moe_expert_major_prefill_test.go:32,48,83`; `metal/prefill_moe_parity_test.go:23-24,107` | none (gate) | n/a | real-MoE paged-off fixture, per-layer cosine, router-flip count, mutation set |
| D-B02 | Major | Host-grouped expert loop: 5 x nE dispatches/layer, per-layer sync, 64-row tile padding; MLX schedules on device with bm=16 | `metal/prefill.go:1345-1353,1400-1428`; header `:36-38` | `quantized.h:2426-2574`, `quantized.cpp:1661-1823,1996`, `utils.h:503-542`, `gather_mm_offsets.metal:8-22` | 5-25% of MoE prefill at 64-128 experts (proj) | count tile waste, then device-side offsets kernel A/B |
| D-P04 | Major | MoE router top-k runs on one GPU thread; recorded about 10% of a fitting ~5 ms MoE token | `metal/moe.go:45-95,97-120,823-832`; Sep 12 `:1438-1440` | none cheaper (`sort.cpp:346-348`) | 4-9% of a resident MoE token (proj on rec) | GPU timestamps around the (1,1) dispatch |
| D-B03 | Major | gpt-oss MXFP4 is requantized to int4 absmax/7; native MXFP4 (E2M1 x E8M0) would keep the checkpoint's values and save 0.6 GB | `decoder/gptoss_safetensors.go:148-192`; aikit `linalg/quant.go:568-640` | `fp_quantized.h:30-38,82-100`, `fp4.h:36-38`, `fp8.h:51-85` | 0.6 GB (cnt); fidelity unquantified | requantization error on the real checkpoint (a CPU-side script) |
| D-B04 | Minor | R18 rows form not applied to shared experts, DeltaNet projections, routed expert GEMVs | `metal/model.go:1455-1480`; `metal/deltanet.go:133-136` | `gemv.h` (R18 source) | 5-12% on shared-expert-heavy/hybrid decode (proj) | wire shared-expert GEMV through `gemvRowsFor` |
| D-P02 | Minor (REVISIT) | M-11 / R11(c) premises are stale for M26 | Sep 12 `:568-590`; `red-october.md:296-305,424` | none | ~0% for the shared-event design (proj) | PROF_SPLIT on aliased v14 M26; N=64 rerun with kill-watch |
| D-B05 | Minor (REVISIT) | `delta_rule` layout: 4x state traffic per token, thread-per-row, uncoalesced; "at ceiling" is stale | `metal/deltanet_kernels.go:128-206`; Sep 12 `:1663` | `gated_delta_update.h:271-366` | 40-96 us/layer recovered (proj) | micro-benchmark lane-per-Dk variant |
| D-P03 | Minor (REVISIT) | 3k expert dispatches per layer vs one batched dispatch; "<1%" premise is paged-only | `metal/moe.go:849-879`; Sep 12 `:1662` | `quantized.h:2087-2209`, `quantized.cpp:1376-1424` | 20 fewer dispatches/layer at k=8 (cnt) | `gemv_w4a8_moe_bench_test.go` with z=k |
| D-C01 | Minor | Expert-major branch hard-wires plain SwiGLU and no biases: wrong for gpt-oss if the feature map ever admits it | `metal/prefill.go:1339-1340,1415-1416`; `metal/model.go:103-119` | none | none today | add FeatAttnSink to a decline test |
| D-D01 | Minor | Ten stale statements (docs and comments) | listed in the entry | none | none | none |
| D-G02 | Minor | `delta_qsplit` / `delta_attn_gate` have no direct test or mutation | `metal/model.go:2964,3037`; `metal/deltanet_test.go:297-301` | none | none | add the two kernels to the chain gate |
| D-N01 | Minor | `touch()` allocates; PROF_SPLIT `os.Getenv` per token | `metal/expertpool.go:430`; `metal/gemma4_moe.go:533` | none | <0.1% (cnt) | none |

### (c) Full entries

#### D-P01 [P] Major: paged MoE prefill is M paged decode tokens

**Claim.** On any paged MoE (M26 and M35 on a 16 GB box are paged by M-13's auto-sizer, `metal/backend.go:214-244`),
prefill declines the batched path and runs M decode tokens. Each token costs 2L+1 synchronous command
buffers and, at N=8, a worst-case 8 x 3.19 MB x 30 layers = 765 MB of expert pread.

**Evidence.**
- Decline: `metal/model.go:1027-1029` (`!(r.moe != nil && r.moe.paged)`, `!m.HasGemma4MoEResident()`,
  `!m.HasPerLayerGeometry()`, `r.dnet == nil`); pinned by `metal/prefill_moe_paged_decline_test.go`.
- Recorded TTFT (rec): alias1 18.7 s, copy2 17.3 s, alias3 17.4 s, copy4 16.8 s on a 96-token prompt, 32 decoded
  tokens (`docs/measurements/m26-alias-fork-collapse-2026-09-24.md:31,177-180`). 96/18.7 = 5.1 and
  96/16.8 = 5.7 prompt tok/s (cnt), against decode 5.98-6.19 tok/s (same table). Prefill runs at the decode rate.
- Linear extrapolation (cnt, from the above): 175-195 ms/token x 512 = 90-100 s; x 8000 = 1,400-1,560 s.
  `red-october.md:309-310` records 1490 s for M35 at 8000 tokens, the same order.
- Command-buffer count (cnt): 512 tokens x 61 = 31,232 per 512-token prompt. An expert-major paged pass with
  N=8 slots would stage at most 8 experts per command buffer: per layer 1 router buffer + ceil(128/8) = 16
  expert buffers, x 30 layers = 510 (worst case, cnt). 61x fewer.
- Bytes (cnt): all-miss per-token = 512 x 765 MB = 392 GB per 512-token prompt. Expert-major reads each
  distinct expert once per layer: at most 128 x 3.19 MB = 408 MB/layer, 12.2 GB per prompt. The CUDA twin
  recorded 28-117 distinct experts per layer at 512 rows (`red-october.md:1608`). A recorded cache hit rate for
  Metal N=8 does not exist; the only recorded hit rate is CUDA's 61.3% at 12 slots (`red-october.md:1608`).
- Pread ceiling used (rec): ~3.7 GB/s at concurrency 1 (`task-never-swap-2026-09.md:741,857`). 12.2 GB / 3.7 GB/s
  = 3.3 s (cnt). The pager's pread is concurrent per layer (`metal/expertpool.go:370-394`), so this is a conservative
  bytes-only floor, not a ceiling on speed.

**Blockers per family (cnt, `metal/model.go:1027-1029`).** Paged alone blocks only the generic paged MoE. M26 additionally
needs `HasPerLayerGeometry` (5 non-uniform layers, `prefill-moe-m26-2026-09-04.md:20`) and
`HasGemma4MoEResident` support in `PrefillLast`. M35 needs batched DeltaNet (D-B01). gpt-oss needs
`FeatAttnSink` in `prefillFeatures` (`metal/model.go:103-119` has none) and correct expert math (D-C01).

**MLX analog.** MLX has no expert pager; it relies on unified-memory page faults. The part to borrow is only
the grouping (D-B02). So this finding is a goinfer design item, not a borrowed technique.

**Projection (proj).** 3-15x TTFT at M=512 on paged M26. Floor 3x: CUDA P20 recorded 2.26-2.66x at M=512 with a
12-slot cache at 61% hit rate (`red-october.md:424,1608`), and an N=8 Metal cache hits less, so the Metal gain
should not be below that. Top: a bytes-only bound of 90-100 s / (3.3 s + about 1 s expert compute counted below)
is about 20x; I state 15x because attention, dense layers and per-layer syncs are unmeasured. Expert compute
(cnt): 2 x 5.95M weights x 8 experts x 30 layers = 2.86 GFLOP/token, x 512 = 1.46 TFLOP, at the recorded R16
rate of ~2.7-2.9 TFLOPS (`metal/prefill.go:26-28`) = about 0.5 s unpadded, about 1 s at 64-row tile padding.

**Fidelity.** Same as D-G01: leaves the exact lane, f16 activations, so it needs the gated lane's evidence on a
MoE, not on the dense 1.5B.

**Probe / kill.** Run `metal/moe_prefill_measure_test.go` with a paged arm at M=64/128/512 on M26 (needs the
fixture, absent here). Kill if paged expert-major is under 2x TTFT at M=512 with N=8, because then pread, not
dispatch count, is the cost and nothing short of a bigger cache helps.

**Why this is not already closed.** Checked Sep 12 M-05 (`audit-metal-2026-09-12.md:248-300`: paged and DeltaNet
families named as unbuilt), `red-october.md:92,307-313` (§2.7 and the row both say "landed 2026-09-17,
unmeasured" for the non-paged path only), and `prefill_moe_paged_decline_test.go`, which pins the decline as
intended behaviour. Nothing schedules the paged build.

#### D-B01 [B,P] Major: no batched DeltaNet prefill

**Claim.** Qwen3.5/3.6/3.8 and Olmo-Hybrid families prefill as M decode tokens because DeltaNet is excluded
from `PrefillLast` (`metal/model.go:1027-1029`). The record's stated reason ("a DeltaNet layer's conv/matrix state must
advance one token at a time", `docs/benchmarks.md:1913-1915`) is a statement about the recurrence only. It
does not preclude batching the projections, MLP and softmax-attention layers, and running only the
recurrence as a token loop inside one kernel.

**What goinfer does (cnt).** 12 dispatches per DeltaNet layer per token (`metal/deltanet.go:129-149`: norm, 4 GEMVs
qkv/b/a/z, conv, gates, L2-norm, rule, gnorm, quantize, out-proj-accumulate). State f32, `[hv, hk]`
per head, read and written inside `delta_rule` (`metal/deltanet_kernels.go:128-206`). Prefill repeats this M times.

**What MLX does (mlx).** `gated_delta_seq` (`kernels/gated_delta_update.h:271-366`) takes q,k,v,g,beta for T tokens
and an f32 `state_in`, loops over T inside the kernel keeping state in registers (`n_per_t = Dk/32` per lane),
and writes `state_out` once. For T>8 a chunked kernel with C=8 (`gated_delta_update.cpp:17-20`, kernel
`gated_delta_update.h:169-265`, `PROCESS_CHUNK_SG` at `:72-167`) restructures the recurrence into the WY form,
so the serial depth is T/8, not T. Supported shapes are Dk=Dv=128 with (Hk,Hv) in {(24,24),(32,32),(16,32),
(16,48),(16,16),(16,64)} (`gated_delta_update.cpp:40-47`); goinfer's test geometry nk=16, nv=48, hk=hv=128
(`metal/deltanet_test.go:71-72`) is in that set. MLX fuses only the recurrence: gates, L2-norm, conv and the
projections stay separate ops (inputs at `fast.cpp:1039-1161`, dispatch `gated_delta_update.cpp:131-209`).

**Delta.** goinfer needs: a T-token causal-conv kernel with ring-state in/out, batched gates and L2-norm
(elementwise, trivial), a T-loop recurrence kernel, then the existing prefill GEMM for qkv/b/a/z/out and the
MLP. DeltaNet layers load no q/k/o weights (`benchmarks.md:1918-1920`), so they would take a new branch in
`PrefillLast`, not the attention branch.

**Recorded base (rec).** Dense 1.5B at K=512: batched 848.0 TTFT tok/s against `--exact-prefill` 83.4
(`metal-prefill-gemm-s2-2026-09-25.md:275,278`) = 10.2x (cnt). That is the ceiling for a fully batched model.

**Projection (proj).** 3-8x TTFT on resident hybrids. Top is below 10.2x because the recurrence stays serial in
T. Floor 3x: if `delta_rule` stays at today's layout, it keeps costing about 3 ms/token on the 35B shape
(Sep 12 `:1663`, rec) while the rest drops by up to 10x, so the speedup is bounded by the recurrence share.
That share is unrecorded for resident hybrids. D-B05 (lane-per-Dk layout) is a prerequisite for a register-state T-loop:
128 f32 per thread in today's thread-per-row layout is too many registers (cnt: hk=128).

**Fidelity.** Moves hybrids from the exact lane (sequential, identical to decode) to the gated lane (f16 activations,
a different reduction order), so it needs its own gate on a hybrid. The state stays f32 in both, so precision of
the carry is unchanged (`fast.cpp` keeps `state_in/out` f32). The chunked WY form is a different algebraic
form, not just a reduction reorder, and needs a long-context cosine check of its own; the T-loop form does not.
The DeltaNet stage gate is cosine >= 0.999999 and maxAbs/rms <= 1e-3 (`metal/deltanet_test.go:35`), not bit-identity.

**Probe / kill.** Build the T-loop kernel plus batched conv/gates at the `metal/deltanet_test.go:71-72` geometry; gate
against the existing chain-vs-CPU test at the same threshold. Kill if the batched hybrid TTFT gain is under 2x at
K=512 on the smallest resident Qwen3.5.

**Why this is not already closed.** `benchmarks.md:1899-1926` (CUDA-side, records the guard as a safety decline),
`red-october.md:313` ("M35 additionally needs a batched Gated-DeltaNet (R11)"), `metal/model.go:1027-1029`. No task or
measurement for a Metal batched DeltaNet exists in the snapshot. `ForwardN` also excludes `r.dnet`
(`benchmarks.md:1920`), so batched verify (area E) is blocked by the same gap.

#### D-G01 [G] Major: expert-major MoE prefill is default ON on a gate that cannot see it

**Status, 2026-10-04: CLOSED, the gate exists and passes.** `TestDG01_expertMajorMoEPrefill` on a real Qwen1.5-MoE layer slice: expert-major against the f16 lane's row-by-row MoE, flips, per-layer K/V and KL; all three planted defects fail it; the shipped path passes at M = 64 and 512 (docs/tasks/task-metal-audit-2026-10.md, "D-G01: PASSES").

**Claim.** The §3.2 pooled gate that licenses the batched f16 lane ran on the dense S model
(`metal/backend.go:611-618`, "S model, K=64/128"; Sep 12 `:85` names it a dense 1.5B). The MoE branch inherits the same
switch. `metal/prefill.go:1314` tests only for `"0"`, so expert-major is ON whenever the lane is.

**Evidence.**
- The MoE tests are `moe_expert_major_prefill_test.go` (fixture `../testdata/mixtral-tiny`, line 32; M in
  {8,16,32,64}, line 48; bar cos(major,seq) >= 0.95, line 83) and `prefill_moe_parity_test.go` (bar 0.95, lines
  22-23, 106, with the explanation that f16 activations versus int8 "mean a high-but-not-exact cosine is expected").
  No test or mutation in the four MoE prefill test files (`grep -i mutat` is empty). The CUDA twin was
  mutation-checked on two fixtures "because a k=2 fixture cannot catch an accumulation-order regression"
  (`red-october.md:1608`).
- The fixture name says Mixtral: softmax top-k, no shared expert, no group routing. `testdata/` is not in the
  snapshot, so I cannot confirm nE/k. At M=64 and top-2 over a handful of experts, an expert gets at most 64 rows,
  so `CePad` never exceeds one 64-row tile (cnt). The multi-tile case (M26 at M=512, mean 32 rows/expert, many experts over
  64) has no test. The untested batched kernels include `moe_route_batch` with sigmoid scoring and
  group-limited routing, the shared-expert branch (`metal/prefill.go:1446-1467`), `shared_gate_add_f16`, and
  `router_gemm_f16` at nE other than the fixture's.
- Not bit-identical to the sequential path in three counted ways: (i) each expert's weighted output is rounded to
  f16 and added into the f16 residual (`scatter_add_weighted_f16`, `metal/prefill.go:829-847`: `residual[...] += half(val)`)
  up to k times, while decode accumulates into the f32 stream in the wacc GEMV (`metal/moe.go:245-273`); (ii) experts
  are applied in ascending expert id (`metal/prefill.go:1434-1444` loop over `byExpert`), decode applies slots in rank
  order j=0..k-1 (`metal/moe.go:849-879`), and float adds do not commute bit-wise; (iii) the router logits come from f16
  activations through `router_gemm_f16` (lane-strided sum then `simd_sum`, `metal/prefill.go:689-709`), while decode
  uses `gemv_wf32_a8` on int8 activations (`metal/moe.go:30-43`). (iii) can change which experts are selected on a
  near tie, a discrete divergence with no cosine bar to catch it.
- G-09 (Sep 12 `:1239-1242`): the MoE prefill measurement exists (`metal/moe_prefill_measure_test.go:30`) and has no
  recorded Metal result; `red-october.md:92` says "unmeasured → measure". Still open.

**Fidelity gate for the lane (the thing that is missing).** Per-layer cosine of the MoE FFN output against the
sequential path on a real MoE checkpoint at M=64/512, a router-flip count (fraction of tokens whose top-k set
differs between the two paths), and one mutation each for: scatter weights, route tie order, shared expert on/off.

**Probe / kill.** The three above, on the smallest real MoE that fits resident. Kill (revert default to opt-in) if
the flip rate exceeds what the f16 dense lane already tolerates, which is itself unrecorded for MoE.

**Why this is not already closed.** `metal/backend.go:611-658` states the gate and its model; `red-october.md:92` says
"landed 2026-09-17, unmeasured"; G-09 is the only record and is open.

#### D-B02 [B,P] Major: device-side expert scheduling instead of host grouping

**Status, 2026-10-04: probed (T1.12, the Qwen1.5-MoE slice).** Host grouping plus the encoding the GPU cannot overlap after each routing sync is 2.4% of a 512-token pass and 1.1% of a 2048-token one. The expert GEMMs run 1.47× the routed rows at M = 512 (1.23× at 2048), which is the lever. The probe also found 14.6 / 77.7 ms of per-call scratch copying before the pass reached the GPU; that is fixed, bit-identical (docs/tasks/task-metal-audit-2026-10.md, "D-B02: T1.12 probed").

**What goinfer does (cnt, `metal/prefill.go:1313-1469`).** Per MoE layer per chunk: batched router, route, `e.End()` and
host readback of `moeIdx`/`moeWgt` (`:1129-1134`), host grouping into `byExpert` (`:1136-1153`), a new encoder,
then per active expert 5 dispatches: gather rows, GU GEMM, SwiGLU, down GEMM, weighted scatter-add
(`:1191-1213`). At nE=128 that is up to 640 dispatches per layer, 19,200 per M26 chunk (cnt), and one
command-buffer boundary per layer (30). Each GEMM launches 64x64 tiles regardless of rows; rows past M stage
zeros and still run MMAs (`metal/prefill.go:38-40`: "What that costs a few-row MoE expert GEMM is NOT measured").

**What MLX does (mlx).** `gather_qmm_rhs` (`kernels/quantized.h:2426-2574`) takes indices already sorted by expert.
`GatherQMM::eval_gpu` selects it when `M == 1 && B >= 16 && right_sorted_ && B / E >= 4`
(`quantized.cpp:1996`). Sorting is a device argsort (`ops.cpp:6307+`, `sorted_indices`). Per-expert offsets come
from a device kernel (`kernels/gather_mm_offsets.metal:8-22`, with `adjust_matrix_offsets` at
`quantized.h:1494-1545`). A device tile scheduler maps threadgroups to (expert, row-tile)
(`kernels/utils.h:503-542`, `schedule_row_tile`). Tiles are bm=16, bn=32, bk=32 (`quantized.cpp:1661-1823`). One
dispatch per projection covers all experts, no host readback. Unsorting is a gather of the [B,k,H] output and a
weighted sum over k.

**Counted waste (cnt).** At M26, M=512, k=8, 128 experts: mean 32 rows per expert. A 64-row tile does 2x the MMA
work of the rows present; at M=128 the mean is 8 rows, 8x. At the recorded R16 rate (2.7-2.9 TFLOPS,
`metal/prefill.go:26-28`) the unpadded expert compute is about 0.5 s per 512-token M26 chunk, so padding is up to
about 0.5 s of extra ALU per chunk. A bm=16 tile cuts the tail waste to under 16 rows per expert (cnt).

**Projection (proj).** 5-25% of MoE prefill time on 60-128-expert shapes. Basis: counted padding (above), counted
dispatch fan-out (19,200 dispatches per chunk, no per-dispatch encode cost is recorded for this encoder), and
30 syncs per chunk. The band is wide because both the encode cost and the 16-row tile's MMA efficiency are
unmeasured. It does not include the paged case, where pread dominates (D-P01).

**Fidelity / bit-identity.** Bit-identical to the current expert-major lane if the R16 inner loop is reused: each
output accumulates K in ordered 8-wide chunks (`metal/prefill.go:40-44`), independent of which rows share a tile, so
changing the tile height or the grouping does not change bits. The reduction over k is the one place bits can
change. Writing per-(token,rank) outputs and summing over rank in order j=0..k-1 in f32, then one f16 round,
would match decode's accumulation order better than today's ascending-expert f16 scatter (G01 (i), (ii)), so
the borrowed structure improves fidelity relative to decode even though it is not bit-identical to the
current prefill lane.

**Probe / kill.** Count `CePad` per active expert on a real MoE to confirm the mean rows; micro-benchmark
`gemm_w4f16_store` at 16, 32, 64 rows on an expert shape; then a device offsets kernel (counting sort, stable) and
one GEMM dispatch with a tile table. Kill if the 16-row tile runs under 70% of the 64-row tile's MMA rate at 32
rows, since then the padding saving is eaten.

**Why this is not already closed.** `metal/prefill.go:38-40` says the few-row cost is not measured; Sep 12 M-05 and
`red-october.md:92` treat expert-major as the end state; no record scopes a device-side scheduler.

#### D-P04 [P] Major: MoE router top-k on one GPU thread

**Status, 2026-10-04: shipped (bit-identical).** The probe measured 35-430 us per (1,1) dispatch, against the 5 us kill line: 68 us at 60x4 and 201 us at 128x8. `moe_route_sg`/`route_gptoss_sg` run on one simdgroup with bit-equal indices and weights. On the Qwen1.5-MoE slice the token is 1.060x faster in-process, 68 us per MoE layer (docs/tasks/task-metal-audit-2026-10.md, "D-P04: SHIPPED").

**Claim.** `moe_route` and `route_gptoss` early-return unless `tid == 0` and are dispatched as (1,1)
(`metal/moe.go:50,101,823-832`). The Sep 12 record notes "moe_route on one GPU thread (deliberate, value-independent
dispatch; ~10% of a fitting ~5 ms MoE token)" (`audit-metal-2026-09-12.md:1438-1440`, rec; the doc names no
measurement for the 10%).

**Counted work (cnt, one thread, nE=128, k=8).** Softmax: max pass, exp pass, scale pass = 3 x 128; `sel` pass 128;
top-k selection k x nE = 1,024 compares: 1,536 loop iterations, over thread-private arrays `score[256]`,
`sel[256]` (2 KB, dynamically indexed, so likely backed by memory, not registers: an inference, not a recorded
fact), once per MoE layer per token. `route_gptoss`: `k x nE` selection with a `taken[]` array, same order.

**Why the stated reason does not require it.** A one-simdgroup dispatch (32,32) is just as value-independent
for encode-ahead as (1,1); the command buffer stays static. There is no MLX analog that is cheaper:
`ArgPartition::eval_gpu` is implemented through the full sort ("We direct arg partition to sort for now",
`sort.cpp:346-348`), so MLX pays more here.

**Delta.** Lane-strided max and exp, k rounds of `simd_max` plus a lowest-index tie-break through `simd_min`
of the index among equal lanes, which reproduces the strict-`>` lowest-index rule at `metal/moe.go:83`.

**Fidelity.** The selection is exact, so the chosen indices are bit-identical. The normaliser sum order is the
only reorder risk: keep the 128-term sum serial in one lane (cheap) and the weights are bit-identical too.
Group-limited routing (`nGroup > 1`) can stay serial.

**Projection (proj).** 4-9% of a resident MoE decode token. Basis: rec 10% of ~5 ms = ~0.5 ms/token; recover
40-90% of it. Applies to resident MoE only; on the paged M26 token (128-167 ms) it is under 1% (proj: 30
layers x ~20 us = 0.6 ms, taking the recorded ~0.5 ms/token spread over an assumed ~24 layers).

**Probe / kill.** GPU timestamps around the (1,1) dispatch on a resident MoE, or a micro-benchmark in the style of
`gemv_w4a8_moe_bench_test.go`. Kill if the dispatch measures under 5 us per layer.

**Why this is not already closed.** N-24 (`audit-metal-2026-09-12.md:1438-1440`) accepted it as deliberate with
no probe; I found no later measurement or comment (`metal/moe.go:44-47` describes the kernel, not its cost).

#### D-B03 [B,P] Major: gpt-oss MXFP4 requantized to int4

**What goinfer does.** Both loaders decode MXFP4 (`mxfp4Pair`) to f32 rows and call `streamQ`, which quantizes
to int4 group-32 f16 scales with `QuantizeGroupInt4Row` (`decoder/gptoss_safetensors.go:148-192`; aikit
`linalg/quant.go:568-640`, scale `maxAbs/7`). Metal then holds int4 and runs `gemv_w4a8_moe*`. The router stays
f32 (`decoder/gptoss_safetensors.go:150`).

**What MLX does (mlx).** `fp_quantized.h` dequantizes E8M0 scales as `fp8_e8m0` for group size 32 (`:30-38`),
and an E2M1 element as `as_type<half>(ushort((bits & 7) << 9)) * 16384.0` (`fp4.h:36-38`). The E8M0 scale
is a bit shift into the f32 exponent (`fp8.h:51-85`). The qdot multiplies activations by the decoded
element and applies the scale once per group (`fp_quantized.h:82-100`).

**Counted difference.** Group size is the same (32), so the requant is a per-group remap. The grids differ: for a
group whose max element is 6 x 2^e (the largest E2M1 magnitude), the int4 scale is 6 x 2^e / 7 and the codes of
{0.5, 1, 1.5, 2, 3, 4, 6} x 2^e become round(7v/6) = {1, 1, 2, 2, 4, 5, 7} (cnt, by the rounding in
`quant.go`). Two pairs collide ((0.5,1) and (1.5,2)); the worst element error is 0.43 x 2^e, 7% of the group
max. Groups with max 4 x 2^e have no collisions but the same 7% worst-case error. gpt-oss's weights are
the QAT'd MXFP4 values, so the checkpoint's own values are the target.

**Illustration (a computation, not a measurement of gpt-oss).** Gaussian weights, 512 x 2880, seed 0, OCP-style
E8M0 scale (floor(log2(amax)) - 2) and nearest-E2M1 rounding, then the `quant.go` rule: int4 vs the MXFP4 values has
10.4% relative RMS per element, 9.9% on a random-x matvec, and only 17.8% of elements survive exactly. Real weights
are not Gaussian and the real checkpoint's scale selection is not in the snapshot, so the real error is unknown.
It is in the same class as any int4 absmax/7 model's noise, but it is extra noise on a model trained
for MXFP4. `capability-matrix.md:137` records "real-oracle 100.0%/0.99843" for gpt-oss without stating the quant
level, so it cannot be read as this requantization's fidelity.

**Memory (cnt).** gpt-oss-20b experts: gate_up 2 x 2880 x 2880 + down 2880 x 2880 per expert (intermediate size
2880 is in `metal/swiglu_quant_gptoss_bench_test.go:11`; 24 layers, 32 experts and hidden 2880 are the public config,
not in the snapshot) = 24.88M weights; x 32 x 24 = 19.1G weights. int4 g32 f16 scale = 4.5 bits, MXFP4 = 4.25 bits
(cnt). 19.1G x 0.25 / 8 = 0.60 GB. The recorded file is 12,109,566,624 B
(`measurements/swap-tripwire-2026-09-22.md:12`), which includes non-expert tensors.

**Native path sketch.** Applying the E8M0 scale needs no multiply: 2^e is an exponent add. E2M1 magnitudes times 2
are the integers {0,1,2,3,4,6,8,12}, which fit int8, so the W4A8 integer dot (int8 activations, int32 accumulate)
survives and the scale applies once per group exactly. Decode of the nibble m (3 magnitude bits): m for m<4,
(2+(m&1)) << ((m>>1)-1) otherwise (cnt, checked for m=4..7), or a 16-entry table. The cost of that unpack against
the existing `UNP8` unpack is unmeasured.

**Fidelity.** A native MXFP4 path would hold the checkpoint's values exactly (E2M1 x 2^e is exact in f16/f32), so
it is strictly closer to the source than the requant. It is not bit-identical to the CPU oracle unless the CPU
path also goes native (`decoder/forward_gptoss.go` uses the requantized weights), and the 0.99843 oracle gate
would move. This is a loader, kernel, aikit and gate change together.

**Probe / kill.** A CPU-side script computing the requant error on the real checkpoint's expert tensors (per-group
max-element distribution, relative RMS, collision rate). Kill if the real error is under 3% relative RMS (below
ordinary int4 noise) and 0.6 GB does not matter for the target fit.

**Why this is not already closed.** `capability-matrix.md:105,137` and `parity-sweep-metal-2026-09-06.md:72` discuss
fit and oracle, not requantization fidelity; `docs/task-mxfp4-gptoss.md` is cited by the bench test at line 11 but is
not in the snapshot. Severity rests on 0.6 GB (>= 500 MB, cnt) plus an unrecorded fidelity cost.

#### D-B04 [B,P] Minor: R18 rows form not extended

**Status, 2026-10-03: shipped (graded the night of 2026-10-03: the 9B token 1.064× at depth 128 and 1.061× at 1024, 7 of 7 reps, bit-identical; `gemvExtOn` on).** Built first, off by default: The
DeltaNet qkv and z GEMVs are the coal family, not the SA family R18's rows kernels reproduce, so "bit-identical by
R18's own argument" needed a new kernel: `gemv_w4a8_coal_staged<R>` (R18's staged down kernel with the coal epilogue).
`TestGemvExt_bitIdentical`: 0 of 6,144 logits differ on the tiny Qwen3.5 hybrid. A one-rep smoke on the Qwen3.5-9B read
the token 1.06× / 1.08× faster at depth 128 / 1024. The routed-expert variant is not built. Grade and rule: the task
doc, "D-B04: built, pending its grade".

R18 (rows-per-simdgroup, 1.17x/1.32x GEMV, bit-identical, `metal-decode-gemv-r18-2026-09-26.md`) is wired at the four
dense decode sites only (`metal/model.go:1455-1480`, `gemvRowsFor` at `:2455`). Not wired: the shared expert (dense GEMVs
inside `encodeMoESharedExpert`, `metal/moe.go:927-939`), the DeltaNet qkv and z projections (`r.pGemv` = `gemv_w4a8_coal`, `metal/model.go:868`, at
`metal/deltanet.go:133,136`; the b/a GEMVs use `pGemvW8`, the out-proj `pSAResid`), and the routed expert GEMVs (`gemv_w4a8_moe`, one row per simdgroup, `metal/moe.go:217-244`). The
shared-expert and DeltaNet sites are a dispatch-site change; the routed GEMVs need a kernel variant. Bit-identical
by R18's own argument.

Projection (proj): 5-12% on shared-expert-heavy resident decode and on resident hybrids. Basis: R18's recorded 1.17x
at its sites applied to the share of the token those GEMVs occupy, which I cannot count without a per-token split.
Severity is Minor because the floor of the band is below 5%; it becomes Major if the share exceeds 30%. Probe: pass
the shared expert through `gemvRowsFor` and A/B on a Qwen2-MoE-style shared-expert fixture. Kill if the gain is
under 1.05x (R18's small-N regime was not characterised).

#### D-P02 [P,D] Minor (REVISIT): M-11 / R11(c) premises

**Premise 1 (M-11, `audit-metal-2026-09-12.md:568-590`; `red-october.md:296-299`):** each of the 61-81 paged command
buffers costs about 14 ms. **Stale:** the M26 paged token takes 128-167 ms (1/7.81 to 1/5.98 tok/s,
`s6-alias-2026-09-24.md:58`; `m26-alias-fork-collapse-2026-09-24.md:177`), and 61 x 14 ms = 854 ms (cnt). The
average per-buffer cost including all work is at most 128/61 = 2.1 to 167/61 = 2.7 ms (cnt). The 14 ms figure
belongs to the earlier swap-bound or CPU-staged state.

**Premise 2 (G-05, `:1154-1159`):** the shared-event verdict was measured where the boundary costs 0.2 ms.
**Partly stale:** the paged boundary is not 14 ms either. The share is unknown, and the token is I/O-dominated:
all-miss pread is 765 MB/token (cnt), 207 ms at the recorded 3.7 GB/s, i.e. 4.8 tok/s, the same order as the
recorded 6.0-7.8 tok/s. Per-buffer sync therefore cannot be the main cost.

**Premise 3 (R11(c), `red-october.md:424`):** the ceiling, the previous default and the floor "have now all failed the
same way; the remaining lever is a lower default". **Stale:** the root cause was a fork-time copy of the MAP_SHARED
mapping, fixed on 2026-09-24 (`m26-alias-fork-collapse-2026-09-24.md:14-18,170-181`); N=8 completes with swap flat
(table `:177-180`, 0 swap delta in 4 of 4 arms). The N=64 and N=32 failures predate that fix and have not been rerun.

**Projection (proj).** The shared-event design saves about 0% of the M26 paged token; I would not build it before the
split below says otherwise. Probe: `GOINFER_MOE_PROF_SPLIT=1` on aliased v14 M26 (`metal/gemma4_moe.go:528-533`) to get
the commit/wait/stage split; and an N=64 rerun under the kill-watch. Kill the REVISIT if staging is under 50% of
the token and commit+wait is over 30%.

#### D-B05 [B,P] Minor (REVISIT): `delta_rule` layout

**Recorded position.** "`delta_rule` about 3 ms/token pinned to the CPU's accumulation order, at ceiling under
the bit-identity contract" (Sep 12 `:1663`); the kernel carries a `CANDIDATE` unroll note measuring
409.8k -> 220k -> 128k ns/dispatch for scalar, 2-wide, 4-wide (`metal/deltanet_kernels.go:144-146`).

**What the kernel does (cnt, `:128-206`).** One thread per (headV, vd) owns a state row of hk=128 f32 (512 B):
pass 1 reads S, decays, writes S, accumulates `kvdot`; pass 2 reads S again, updates, writes S, accumulates `o`.
Per token the state is read twice and written twice, and the 32 lanes of a simdgroup touch 32 rows 512 B apart on every
step (thread-row layout, no coalescing).

**What MLX does (mlx).** Lane-per-Dk: `n_per_t = Dk/32` elements per lane in registers, `simd_sum` over Dk, state
read once and written once (`gated_delta_update.h:271-366`), state f32 (`fast.cpp:1039-1161`).

**Floor (cnt, rec).** State 2 MB/layer (rec, Sep 12 `:1663`); 1 read + 1 write = 4.2 MB, at 200 GB/s = 21 us per
layer. The recorded kernel time is about 128 us/dispatch (rec, comment above; the shape of that measurement is not
stated). Today's kernel is about 6x the floor; the gap is about 100 us/layer. Over 30 DeltaNet layers (assumes a 3:1
40-layer M35, not in the snapshot) the recoverable amount is at most 3.0 ms/token, i.e. 0.6% of the 457 ms
paged M35 token (rec 2.19 tok/s, Sep 12 `:1708`).

**Why "at ceiling" is stale.** The stage gate is cosine >= 0.999999 / rel 1e-3 (`metal/deltanet_test.go:35`), so the
accumulation order is a design choice mirroring CUDA, not a gate requirement, and the 4-wide result in the comment
shows the kernel is far from a bandwidth bound. A lane-per-Dk layout changes the summation order (`simd_sum`), so it
is not bit-identical to the current kernel. That decision is Francis's.

**Projection (proj).** 40-96 us/layer recovered (basis: 40-100% of the 96 us gap to the counted floor). Minor on the paged
M35 token; relevant to D-B01 (register-state T-loop) and to resident hybrid decode, where the token time is
unrecorded. Probe: micro-benchmark a lane-per-Dk variant at hk=hv=128, nv=32 and 48. Kill if it is not at least 2x.

#### D-P03 [P] Minor (REVISIT): one dispatch over the k selected experts

**Status, 2026-10-04: parked, off by default (bit-identical).** Four dispatches instead of 3k, with an exact slot-order combine. On the Qwen1.5-MoE slice (k = 4) the token is 1.019x faster in-process, 21.5 us per MoE layer: the owner's park zone. A k = 8 model is unmeasured (docs/tasks/task-metal-audit-2026-10.md, "D-P03: PARKED").

**What goinfer does.** `encodeMoEExperts` loops j over k and issues GU GEMV, SwiGLU, down-wacc each time
(`metal/moe.go:849-879`): 3k = 24 dispatches per layer at k=8 (cnt), plus the router and shared expert. The slot is a
uniform index into `rIdx` (`mo.uSlot[j]`).

**What MLX does (mlx).** `gather_qmv` takes an index buffer and runs all (token, slot) pairs in one dispatch over the
z grid (`quantized.h:2087-2209`, `quantized.cpp:1376-1424`). It reads each expert once per (token, slot) pair,
with no cross-token dedup, which is the same read pattern as goinfer's per-slot dispatches. So the gain is
dispatch count, not bytes.

**Premise.** "batching across k saves <1% of the paged token" (Sep 12 `:1662`). True for the paged token (I/O-bound,
D-P02) and false as a general statement: resident MoE has no I/O term.

**Counted.** With a z=k GU dispatch, one batched SwiGLU, one z=k down dispatch writing per-slot outputs and one
combine kernel, the layer needs 4 dispatches instead of 24: 20 fewer (cnt). At the recorded ~3.8 us launch floor
(rec, Sep 12 `:1662`) that is 76 us/layer. The resident MoE token time is not recorded, so I cannot give a fraction.

**Fidelity.** The down projection accumulates into the f32 stream one slot at a time (`mo.pDownWacc`); z-parallel
threadgroups would race on that target. A combine kernel adding the k weighted outputs in j order, with the same
float ops, is bit-identical; atomics would not be. GU overwrites one shared `r.gu` today, so it needs k scratch slots.

**Probe / kill.** `gemv_w4a8_moe_bench_test.go` with a z=k variant at a resident MoE shape. Kill if the saving is
under 2% of the layer.

#### D-C01 [C] Minor: latent wrong gpt-oss expert math in the expert-major branch

The expert-major branch dispatches `route_gptoss_batch` for gpt-oss (`metal/prefill.go:1339-1340`), then the plain
`pSw` SwiGLU with `r.uAct` (`:1200-1201`) and down GEMM with `dummyBias`: no clamped interleaved SwiGLU, no per-expert
gate/up and down biases. Decode uses `swiglu_quant_gptoss` and `gemv_w4a8_moe_wacc_bias` (`metal/moe.go:869-872`). Not
reachable today: `prefillFeatures` has no `FeatAttnSink` (`metal/model.go:103-119`), so `MissingResidentFeatures` makes
`prefillOK` false for gpt-oss (`metal/model.go:1027`). The route dispatch suggests someone expected it to be reachable.
The fix is a decline test on `isGptOss` inside the expert-major branch, so a future feature-map edit fails closed.

#### D-D01 [D] Minor: stale statements

1. `red-october.md:296-305` (§2.6): "61-81 sync command buffers at ~14 ms each", "counted ceiling 3-4 tok/s from
   ~2" (see D-P02: M26 N=8 already measures 6.0-7.8 tok/s; that ceiling was a 35B figure).
2. `red-october.md:424` R11(c): three swap spirals, "remaining lever is a lower default" (root cause fixed 2026-09-24).
   Also the row at `:91` ("2L+1 sync command buffers at ~14 ms each ... 3-4 tok/s counted").
3. `red-october.md:1608` R11(b) says bit-identical (CUDA's P20). The Metal row (`:92`) is a different lane; read-across
   would be wrong (D-G01).
4. `benchmarks.md:1976-1985`: "M35/M26 off-limits on this box on any path"; `:2017` "M26 on Metal is still off-limits":
   the 2026-09-24 M26 Metal runs exist (`m26-alias-fork-collapse-2026-09-24.md`).
5. `benchmarks.md:2260-2262` (and `:1964-1966`, which is correct): "no resident Metal backend for gpt-oss" while
   `metal/model.go:140,343` and `capability-matrix.md:105` say declared and resident.
6. Sep 12 §5 `:1665-1666`: "`NewBufferNoCopy` inapplicable to goinfer's fused/narrowed layouts", superseded by S6
   (723 MB aliased on M26: 60 single tensors + 55 fused groups, `s6-alias-2026-09-24.md:55-56`).
7. `metal/model.go:95-101`: "the FFN half runs ROW BY ROW", true only for the fallback and the paged path.
8. `metal/prefill_moe_parity_test.go:7-24`: same row-by-row description of the path under test.
9. `metal/gptoss_kernels_test.go:8-9`: "FeatAttnSink is still not declared for Metal".
10. `metal/deltanet_test.go:297-301`: "delta_qsplit/delta_attn_gate ... neither has Go-side wiring yet", wired at
    `metal/model.go:2964,3037`. Also `metal/deltanet_kernels.go:144-146`: the `CANDIDATE` comment sits in shipped code and states
    a test whose result is not in the snapshot.

#### D-G02 [G] Minor: two DeltaNet kernels with no direct gate

`delta_qsplit` and `delta_attn_gate` (`metal/deltanet_kernels.go:235-246`) are wired (`metal/model.go:2964,3037`) but not in the chain
gate (`metal/deltanet_test.go:297-301`) and have no mutation. `qwen35_resident_parity_test.go` covers them end to end only,
which cannot localize a wrong q/gate split.

**Status, 2026-10-01.** Done, beside the chain gate (tag `goinfer_testhooks`): `TestQGateKernels_cpuParity` runs
`delta_qsplit` then `delta_attn_gate` as `encodeLayer` launches them, over three geometries with gate values out to ±100,
against the CPU's own split and gate. Those are now `splitQGate` and `qGateContext` in `decoder/forward_qwen35.go`,
called by both CPU paths and exported as test hooks. The split matches exactly and writes nothing past its length; the
gate factor is within 2.1e-7 of the CPU's (bar 1e-6). `TestQGateKernels_mutations`: swapped halves, two blocks instead
of per-head interleaving, no sigmoid, and the sigmoid's sign flipped each fail it.

#### D-N01 [N] Minor

`metal/expertpool.go:430` `touch()` allocates on a path run per expert per layer per token; `metal/gemma4_moe.go:533` reads
`GOINFER_MOE_PROF_SPLIT` through `os.Getenv` per token. Both are under 0.1% of a 128 ms token (cnt: 30 layers x 8 touches
x ~100 ns). N-23's dense-layer batching in `metal/gemma4_moe.go:606-612` is moot for M26 (30 layers, all MoE).

#### Answers to the assigned questions

1. **MLX gather GEMV reads each expert once per (token, slot), with no cross-token dedup.** One dispatch covers all
   pairs (`quantized.h:2087-2209`; `quantized.cpp:1376-1424`). goinfer's `gemv_w4a8_moe` has the same read pattern
   in k separate dispatches (`metal/moe.go:849-879`): same bytes, more dispatches (D-P03).
2. **Expert-major batching** is MLX's sorted `gather_qmm_rhs` (D-B02). The Sep 12 record already scoped it (M-05) and
   goinfer built it for non-paged generic MoE (`metal/prefill.go:1313-1469`, default ON); it left paged, Gemma-4 MoE,
   DeltaNet and gpt-oss families on per-token decode (D-P01).
3. **DeltaNet cost** is dominated by `delta_rule`'s access shape and by per-token repetition in prefill, not by dispatch
   count or intermediate buffers: 12 dispatches/layer but 2 MB of f32 state traffic and no readback (Sep 12
   `:1663`). MLX fuses only the recurrence; it does not fuse conv, gates, L2-norm or gnorm (`fast.cpp:1039-1161`).
   What MLX has that goinfer lacks: the T-loop kernel and the C=8 chunked kernel (D-B01, D-B05).
4. **Recurrent-state precision.** f32 in both (`fast.cpp` state_in/out f32; goinfer `D.state` float). The goinfer
   stage gate is cosine >= 0.999999, not bit-identity (`metal/deltanet_test.go:35`); the bit-identity language in
   `metal/deltanet_kernels.go:144-146` is a self-imposed rule that mirrors CUDA.
5. **Paged status after S3-S6.** M26 N=8 is stable (swap flat in 4 of 4 post-fix arms, 5.98-6.19 tok/s, TTFT 16.8-18.7 s;
   alias and copy arms equal within noise; `m26-alias-fork-collapse-2026-09-24.md:175-181`). Open: sequential prefill
   (D-P01), N>8 rerun, M35 on Metal post-S6 (unrecorded), the shared-event half (D-P02).

### (d) Carry-forward (Sep 12 IDs in my area)

| ID | Status | Evidence |
|---|---|---|
| M-05 (MoE prefill rows; paged/DeltaNet as M decode tokens) | PARTIAL | Non-paged generic MoE built and default ON (`metal/prefill.go:1313-1469`, `:1101`). Paged OPEN (`metal/model.go:1027-1029`, D-P01), DeltaNet OPEN (D-B01), Gemma-4 MoE OPEN |
| M-07 (host copies; my part: paged/giw) | PARTIAL, superseded in part | S6 NoCopy alias default ON; M26 723 MB aliased (`s6-alias-2026-09-24.md:55-56`). Not re-audited here (area C) |
| M-11 (61-81 command buffers at ~14 ms) | PARTIAL + REVISIT | Contiguous pool + `slotIdx` shipped (`metal/moe.go:881-921`, `metal/expertpool.go:213-216`). Shared-event half open; premise stale (D-P02) |
| M-12 (concurrent staging) | CLOSED-VERIFIED | Cross-expert goroutines `metal/expertpool.go:370-394`; per-expert spans `metal/moe.go:673-687` (3), `metal/gemma4_moe.go:356-360` (2). Wall-clock win not isolated, but 6.0-7.8 tok/s recorded |
| M-13 (auto-sized slots) | CLOSED-VERIFIED | `autoMoESlots` `metal/backend.go:214-244`, floor `moeTopK`, ceiling 64 |
| M-14 (residency set per command buffer) | CLOSED-VERIFIED | `e2.UseResidencySet` `metal/moe.go:1016`, `metal/gemma4_moe.go:594`; rationale `metal/model.go:1639-1643` |
| C-02 (HiddenLast/Forward/ForwardArgmax bind zero stacked buffers on paged) | CLOSED-VERIFIED | `metal/backend.go:590-593,885` decline to the head-bearing/paged path |
| G-05 (shared-event verdict on a shape without the cost) | OPEN, superseded | No paged re-run; see D-P02 |
| G-09 (MoE prefill measurement unrun) | OPEN | `metal/moe_prefill_measure_test.go:30` exists, no recorded Metal result; `red-october.md:92` "unmeasured" |
| N-20 (per-stage scale re-derivation) | CLOSED-VERIFIED | `int4DirectBytesOnly` use at `metal/moe.go:600-602`, `metal/gemma4_moe.go:292-293` |
| N-21 (zero-init of slots) | CLOSED-VERIFIED | `gpu.NewBufferLenOf` `metal/expertpool.go:213-216` |
| N-22 (merge phase 2 / phase 1) | superseded by M-11 | Same status as M-11 |
| N-23 (dense layers' own Begin/End) | CLOSED in `moe.go`; OPEN and moot in `gemma4_moe.go` | `metal/gemma4_moe.go:606-612` per-dense-layer buffers; no dense layers in M26. Mixed dense+paged path has no fixture (Sep 12 `:1418-1436`) |
| N-24 (f32 router weight; `moe_route` on one thread) | f32 weight NEGATIVE-CLOSED (deliberate, <= 0.4 ms); single-thread route REVISIT | D-P04 |
| N-36 (KV charged for DeltaNet layers) | CLOSED | `decoder/residentneed.go:61` per Sep 12 `:1562` (read as recorded; not re-derived) |
| Sep 12 §5 items (paged internals) | CLOSED-VERIFIED | See (e) |

### (e) Checked and found correct

- Router cap backstop: `metal/moe.go:483` rejects nE or nGroup over `ResidentBackendMoECap`, matching the `score[256]`, `gscore[64]`
  arrays in `moe_route` (`metal/moe.go:50-95`).
- Paged equals stacked: the byte-identity trick and the `TestMoEPagingPread_matchesByteCopy` / `TestGemma4Paging_bitExact`
  gates exercise eviction and restaging (Sep 12 `:1657-1661`; tests present in `moe_paging_test.go`).
- `metal/gptoss_kernels_test.go:626-674` AST guard on per-expert bias indexing is a real discriminator.
- `scatter_add_weighted_f16` has no write race within a dispatch: a token appears at most once per expert (distinct
  top-k), and dispatches in one encoder run in order. `CePad` row masking and scratch sizing: rows past `Ce` are zeroed
  on gather and not scattered (`metal/prefill.go:829-847`, gather at `:1191-1194`).
- `rowIdxBuf`/`rowWgtBuf` are filled after `e.End()` on the host and read by a later encoder, so the ordering is safe
  (`metal/prefill.go:1345-1353,1358-1399`).
- Abort discipline on `e.Err()` after both command buffers (`metal/prefill.go:1350-1353`).
- `moe_route`'s selection is exact and tie-stable (strict `>`, lowest index, `metal/moe.go:83`), and cheaper than MLX's
  sort-based argpartition by shape (not measured).
- DeltaNet f32 state parity with MLX; `delta_gnorm` normalise-then-gate ordering and the `sqrt(1/(ss+eps))` zero guard are
  documented and tested (`metal/deltanet_kernels.go:15-30`, `TestDeltaNorm_zeroHead`).
- The `canUseF16Lane` scope excludes MoE and DeltaNet explicitly (`metal/model.go:2670-2673`), so R1's lane cannot leak into them.
- R18 is not claimed beyond its four sites (`metal/model.go:1455-1480`); I report the unwired sites in D-B04 only.

### (f) Not settleable statically

1. PROF_SPLIT split (staging vs commit vs wait) on aliased M26 at N=8: decides D-P02.
2. N=64 and N=32 reruns under the kill-watch after the fork-collapse fix; M35 on Metal post-S6.
3. Metal expert-major speedup and fidelity on a real MoE (G-09), router-flip rate, and `testdata/` contents (nE, k,
   shared expert in `mixtral-tiny`; the fixtures are not in the snapshot).
4. 16-row against 64-row tile cost on an expert shape (D-B02 kill criterion); per-dispatch encode cost.
5. `delta_rule` gain from a lane-per-Dk layout on M1 Pro; the resident-hybrid token breakdown (recurrence share).
6. Real gpt-oss requantization error; cost of an E2M1 integer unpack against `UNP8`.
7. Resident MoE decode time (no recorded figure beyond "~5 ms fitting"): sizes D-P03, D-P04, D-B04.
8. Whether any registered generic-MoE family also declares `FeatSandwichNorm` or `FeatPostOnlyNorm`: neither the decode MoE
   chain (`metal/moe.go:791-798`) nor either prefill MoE branch applies the post-MLP norm (`metal/prefill.go:1469-1490` is the dense path
   only).
9. Chunked WY form cosine against the sequential recurrence over a long context.


## 8. Area E — Batched decode (MC3), speculative verify, multi-slot KV, chunked prefill, quantized KV, LoRA, sampling

Reviewer E. goinfer `844700f8`, aikit v1.51.0, MLX `9c3d355`. Static review only. Tags: [rec] a figure the repo recorded
(doc:line), [cnt] counted from code shape, [mlx] MLX source (file:line in mlx/backend/metal), [proj] a
projection with its band and basis.

### (a) Summary

1. The batched step is the right shape for the 1.5B at B>=3 and wrong-shaped in two places on the 7B. The 1.5B step's
   matmuls cost 14.2-14.5 ms at any B (concurrency-mc3-s3:21-26 [rec]); the fragment kernels are ALU-bound (S4:14-27 [rec]),
   so cost per token falls 4.5x from B=1 to B=8 (10.46 ms -> 2.34 ms [rec, counted]). On the 7B the fragment is 2.7x slower than
   the production GEMV per call (1.167 against 0.435 ms, S0:92-103 [rec]), the shipped build instantiates FB=2 where S0 recorded
   FB=4 as best for gate|up and down, and B=2 runs per-row GEMVs that each stream the full weights.
2. Largest unclaimed lever: the 8-row step is a bit-identical exact prefill engine. A fresh prompt under the 64-token floor
   still runs one token at a time (305 ms at K=32 [rec via area A]); the same tokens in 8-row same-slot pieces cost ~74 ms
   [proj from two recorded ratios]; the 7B ~1.06 s -> ~0.28 s [proj]. No gate is needed because the step is bit-identical to sequential decode. E-P01, Major.
3. E-X01 (cross-area) was raised as a possible all-or-nothing item: a 2026-09-28 record said the `--embed-int4` default made the
   Metal resident decline to the CPU. **Resolved at consolidation:** the working repo's `loadflags` defaults it off on
   `-backend metal`, so a default Metal serve stays resident (§3 note 1). The entry below is kept as the reviewer wrote it.
4. MLX's attention does not do what `mc3_attention_rows` does better: `sdpa_vector` takes one key count for the whole
   dispatch and expresses per-batch lengths only through a mask (scaled_dot_product_attention.cpp:951-1004 [mlx]).
   goinfer's per-row `nKeys` is ahead of it. What goinfer lacks is the other half: rows at attention_fa depth keep two
   dispatches each, serially, so deep rows do not amortize (E-P05).
5. Sampling: temperature-only requests are on-device and reproducible (Gumbel/Philox, 0 mismatches in 15,840 [rec]). Any
   top_p/top_k/min_p request still takes the host full-row path, counted at ~13% of a 1.5B token (E-P04).
6. Correctness review found no wrong-output path. Slot ordering, stale KV on slot reuse, the sampled-token join and Gumbel
   reproducibility all hold on the code read (section e). Two latent items: the solo path returns a reused logits buffer
   (E-C01) and the Gumbel pragma bracket interacts with `GOINFER_PRECISE_MATH` (E-C02).
7. The headline contract (bit-identical to unbatched decode) has no default-run test: every MC3/MC5/spec-step identity test
   skips unless `GOINFER_METAL_MC3=1` and a real checkpoint is present (E-G01).
8. Carry-forward: M-08, C-03, C-04, N-26 CLOSED-VERIFIED; N-27 NEGATIVE-CLOSED; N-28 PARTIAL; N-03 still holds; "batched
   small-M verify" superseded (shipped); "fused argmax" stays a recorded negative with one REVISIT.

### (b) Findings table (ordered by effect)

| ID | Sev | Claim | Evidence | MLX analog | Band | Probe |
|---|---|---|---|---|---|---|
| E-X01 | **Resolved at consolidation** (was: Major if confirmed, cross-area) | Default `-embed-int4` was suspected of making the Metal resident decline to CPU on a default serve; `internal/loadflags/loadflags.go:130-139` defaults it off on `-backend metal` (§3 note 1) | mc3-prefill-attr-2026-09-28.md:27; metal/model.go:520-526,1353-1363; decoder/model.go:382-388 (comment says default off); task-never-swap-2026-09.md:281-283 | none | all-or-nothing | `goinfer serve` on the Mac; read the `decode path:` banner line (internal/serveapp/banner.go:112) |
| E-P01 | Major | Prompts below the 64-token floor, and short reuse suffixes, run sequentially or in a >=96 ms batched pass; the bit-identical 8-row step does them at ~2.3 ms/token | metal/backend.go:626,772-775; decoder/model.go:1583-1604; metal/batch.go:772-803; metal-spec-step-verify-2026-09-27.md:35-41 | quantized.cpp:89-130 (qmv to M<14 on M1 Pro) | K=32 TTFT 305 -> ~74 ms [proj]; 7B ~1.06 s -> ~0.28 s [proj] | prefill a 32-token prompt via `forwardMultiInto` pieces; compare bytes and time |
| E-P02 | Minor (Major if probe lands) | 7B at B=2 runs per-row GEMVs that each stream the whole gate\|up weight; MLX launches the M rows adjacent so they share weight reads | metal/batch.go:605-619,640-645; quantized.cpp:495-496; kernels/quantized.h:787-794 | qmv grid (M, N/bn) | 0-15% of the 7B B=2 step [proj], ceiling 22% | standalone 2-row gate\|up on the 7B shape |
| E-P03 | Minor (7B) | FB=2 shipped for `mc3_bt`/`mc3_btd`; S0 recorded FB=4 best on the 7B for gate\|up and down, 49 of ~66 ms | metal/batch.go:140,205,366; metal/gemm_mma8_mc3_test.go:223-224; concurrency-mc3-s0:92-103 | none (register-resident fragments) | 0-8% of the 7B B>=3 step [proj] | instantiate fb4, run `TestMC3Step_throughput` on the 7B |
| E-P04 | Minor (Major if top_p clients count as default) | No Metal device top-K/top-p/min-p; filtered requests host-select over 152k logits | red-october.md:90; benchmarks.md:1104; decoder/model.go:1971-1980 | none fused (sort.cpp only) | ~13% of a 1.5B token [cnt] | filtered sampled cell on the Mac |
| E-P05 | Minor | Rows at attention_fa depth run 2 serial dispatches each; the depth-related cost per extra row is 0.085 tokens at 2048 and the batched step has no timing above depth 1536 | metal/batch.go:621-637; metal-spec-step-verify:35-41; concurrency-mc3-s3:63 | sdpa_vector_2pass: all q rows of a kv-head in one TG (sdpa_vector.h:180+; scaled_dot_product_attention.cpp:772,817) | 5-12% of an 8-row step at depth 2048 [proj]; ceiling 21-25% [cnt] | `TestMC3StepBreakdown` at 2048/4096, B=8 |
| E-P06 | Minor | One 7B@2048 verify cost curve for every model; the 1.5B at depth 128 costs 12/23/51% less at 2/4/8 rows, so the controller declines paying drafts | metal/batch.go:749-752; spec_adaptive.go; spec-step-verify:35-41 | none | 0-5% of 1.5B chat tok/s [proj] | calibrate the curve at load, like `calibrateRows` |
| E-P07 | Minor | Qwen3-family dense models (QK-norm) are outside MC3: no batching, no step-kernel verify | metal/batch.go:311-312; metal/kernels.go:2010-2025; benchmarks.md:508 | n/a | MC3 gave the 7B 4-client aggregate 1.785x the old build (concurrency-mc3-7b-w7:74 [rec]); Qwen3 forgoes it | derive `mc3_qk_norm_rows` with the `edit` method |
| E-P08 | Minor | Greedy rows in a batched step copy 607,744 B each and argmax on the host | metal/batch.go:691-709; metal/model.go:1955-1969 | none | <=2.9% at B=8, <=1.4% at B=4 [cnt] | REVISIT of "fused argmax" |
| E-G01 | Minor (Major if CI lacks the env) | MC3/MC5/spec-step identity gates are opt-in: env var plus a real checkpoint | gemm_mma8_mc3_test.go, mc3_step_test.go, mc5_chunk_test.go, spec_verify_identity_test.go (skip lines) | n/a | n/a | read CI config (absent from snapshot) |
| E-C01 | Minor, latent | MC3 solo path hands out `r.logitsHost`, a buffer the next generation's solo step rewrites | metal/backend.go:557-559; metal/model.go:1955-1969; decoder/model.go:2175-2182 | n/a | no output change today | two generations, one with a 30 ms `LogitProcessor` |
| E-C02 | Minor, default-off | `fp contract(fast)` restore after the Gumbel block also covers `mc3RowsKernels` under `GOINFER_PRECISE_MATH` | metal/gumbel.go:49,164; metal/model.go:846-851 | n/a | n/a | `GOINFER_PRECISE_MATH=1` with `TestMC3Step_bitIdentical` |
| E-X01 | **Resolved at consolidation** (was: Major if confirmed, cross-area) | Default `-embed-int4` was suspected of making the Metal resident decline to CPU on a default serve; `internal/loadflags/loadflags.go:130-139` defaults it off on `-backend metal` (§3 note 1) | mc3-prefill-attr-2026-09-28.md:27; metal/model.go:520-526,1353-1363; decoder/model.go:382-388 (comment says default off); task-never-swap-2026-09.md:281-283 | none | all-or-nothing | `goinfer serve` on the Mac; read the `decode path:` banner line (internal/serveapp/banner.go:112) |
| E-P01 | Major | Prompts below the 64-token floor, and short reuse suffixes, run sequentially or in a >=96 ms batched pass; the bit-identical 8-row step does them at ~2.3 ms/token | metal/backend.go:626,772-775; decoder/model.go:1583-1604; metal/batch.go:772-803; metal-spec-step-verify-2026-09-27.md:35-41 | quantized.cpp:89-130 (qmv to M<14 on M1 Pro) | K=32 TTFT 305 -> ~74 ms [proj]; 7B ~1.06 s -> ~0.28 s [proj] | prefill a 32-token prompt via `forwardMultiInto` pieces; compare bytes and time |
| E-P02 | Minor (Major if probe lands) | 7B at B=2 runs per-row GEMVs that each stream the whole gate\|up weight; MLX launches the M rows adjacent so they share weight reads | metal/batch.go:605-619,640-645; quantized.cpp:495-496; kernels/quantized.h:787-794 | qmv grid (M, N/bn) | 0-15% of the 7B B=2 step [proj], ceiling 22% | standalone 2-row gate\|up on the 7B shape |
| E-P03 | Minor (7B) | FB=2 shipped for `mc3_bt`/`mc3_btd`; S0 recorded FB=4 best on the 7B for gate\|up and down, 49 of ~66 ms | metal/batch.go:140,205,366; metal/gemm_mma8_mc3_test.go:223-224; concurrency-mc3-s0:92-103 | none (register-resident fragments) | 0-8% of the 7B B>=3 step [proj] | instantiate fb4, run `TestMC3Step_throughput` on the 7B |
| E-P04 | Minor (Major if top_p clients count as default) | No Metal device top-K/top-p/min-p; filtered requests host-select over 152k logits | red-october.md:90; benchmarks.md:1104; decoder/model.go:1971-1980 | none fused (sort.cpp only) | ~13% of a 1.5B token [cnt] | filtered sampled cell on the Mac |
| E-P05 | Minor | Rows at attention_fa depth run 2 serial dispatches each; the depth-related cost per extra row is 0.085 tokens at 2048 and the batched step has no timing above depth 1536 | metal/batch.go:621-637; metal-spec-step-verify:35-41; concurrency-mc3-s3:63 | sdpa_vector_2pass: all q rows of a kv-head in one TG (sdpa_vector.h:180+; scaled_dot_product_attention.cpp:772,817) | 5-12% of an 8-row step at depth 2048 [proj]; ceiling 21-25% [cnt] | `TestMC3StepBreakdown` at 2048/4096, B=8 |
| E-P06 | Minor | One 7B@2048 verify cost curve for every model; the 1.5B at depth 128 costs 12/23/51% less at 2/4/8 rows, so the controller declines paying drafts | metal/batch.go:749-752; spec_adaptive.go; spec-step-verify:35-41 | none | 0-5% of 1.5B chat tok/s [proj] | calibrate the curve at load, like `calibrateRows` |
| E-P07 | Minor | Qwen3-family dense models (QK-norm) are outside MC3: no batching, no step-kernel verify | metal/batch.go:311-312; metal/kernels.go:2010-2025; benchmarks.md:508 | n/a | MC3 gave the 7B 4-client aggregate 1.785x the old build (concurrency-mc3-7b-w7:74 [rec]); Qwen3 forgoes it | derive `mc3_qk_norm_rows` with the `edit` method |
| E-P08 | Minor | Greedy rows in a batched step copy 607,744 B each and argmax on the host | metal/batch.go:691-709; metal/model.go:1955-1969 | none | <=2.9% at B=8, <=1.4% at B=4 [cnt] | REVISIT of "fused argmax" |
| E-G01 | Minor (Major if CI lacks the env) | MC3/MC5/spec-step identity gates are opt-in: env var plus a real checkpoint | gemm_mma8_mc3_test.go, mc3_step_test.go, mc5_chunk_test.go, spec_verify_identity_test.go (skip lines) | n/a | n/a | read CI config (absent from snapshot) |
| E-C01 | Minor, latent | MC3 solo path hands out `r.logitsHost`, a buffer the next generation's solo step rewrites | metal/backend.go:557-559; metal/model.go:1955-1969; decoder/model.go:2175-2182 | n/a | no output change today | two generations, one with a 30 ms `LogitProcessor` |
| E-C02 | Minor, default-off | `fp contract(fast)` restore after the Gumbel block also covers `mc3RowsKernels` under `GOINFER_PRECISE_MATH` | metal/gumbel.go:49,164; metal/model.go:846-851 | n/a | n/a | `GOINFER_PRECISE_MATH=1` with `TestMC3Step_bitIdentical` |
| E-P09 | Minor (Major on the 7B if pages count) | Default 4 KV slots at 4096: 3 extra slots are ~351 MB (1.5B) and ~705 MB (7B) | decoder/fitplan.go:184; internal/serveapp/main.go:431; metal/backend.go:377-417; concurrency-mc1:24 | n/a | memory only | resident-set after load, 1 vs 4 slots |
| E-P10 | Minor, opt-in | `kv_store_i8` is nKV one-thread threadgroups with a serial 128-iteration loop; `--kv i8` forfeits MC1/MC3/spec verify | metal/kernels.go:907-929; metal/model.go:3006-3011; metal/batch.go:311 | n/a | ~1% [cnt, unmeasured] | micro-bench |
| E-N01 | Minor | 112 `pack` dispatches per step could be fused into their producers | metal/batch.go:609,645,649 | n/a | <=0.9% [cnt] | none worth running first |
| E-D01 | Minor | `--spec` help says "on the CPU backend"; `Options.EmbedInt4` comment says default off | internal/serveapp/main.go:453; decoder/model.go:382-388 | n/a | n/a | edit text |

### (c) Full entries

#### E-X01 [resolved] Default `-embed-int4` and the Metal resident (cross-area)

**Resolved at consolidation (2026-09-30).** `internal/loadflags/loadflags.go:62-74` registers `-backend` with default `cpu` (`auto` since R17, 2026-10-01) and sets
`EmbedInt4 = true`; `:122-131` (`embedInt4()`) returns false when `Backend == "metal"` and `--embed-int4` was not given on the command
line, with a comment recording the same fault (found 2026-09-28, reproduced 2026-09-30 on the 0.5B). An explicit `--embed-int4` on Metal
is honoured and runs on the CPU as asked. The text below is the reviewer's original reasoning, from before `loadflags` could be read.

Metal requires an int8 LM head: `int8BufA` falls to `int8Buf`, which errors on a non-int8 weight (metal/model.go:520-526),
and the head load is at metal/model.go:1358-1368. The in-snapshot comment on `Options.EmbedInt4` says "default off keeps the
bit-exact int8 pin" (decoder/model.go:382-388) and `embeddingWith` agrees (decoder/weightmat.go:95-106). But
task-never-swap-2026-09.md:281-283 records an owner decision of 2026-09-28 "to make `--embed-int4` the default", and
mc3-prefill-attr-2026-09-28.md:27 reads: "Since 9ccf7fb1 the int4-embedding default makes the Metal resident decline to the
CPU; a cell whose log does not show `decode path: metal-resident` is void." That run passed `-embed-int4=false`
explicitly. The default itself lives in `loadflags` (internal/serveapp/main.go:180 reads `cfg.load.Options()`), which is not in the
snapshot. B-N01 reads the same error as "Metal cannot load an `-embed-int4` bundle" and rates it Minor; if the default is
now on, the same fact is a default-path fallback to CPU for every Metal user.
Probe: run plain `goinfer serve` on the Mac and read the banner's `decode path:` line. Not settleable statically.
Severity: Major if the default is on and the decline is silent to the user; the banner does print the path.

#### E-P01 [P, Major] The 8-row step is an exact prefill engine for short prompts

What happens today. `residentPrefillSeed` sends a suffix of 8 tokens or more to `PrefillLast` (decoder/model.go:1583-1604).
`metalResident.PrefillLast` declines when `startPos+len < metalFastPrefillFloor` (64) with an error (metal/backend.go:782-797), the
decoder warns and runs the sequential loop, with `ForwardNoLogits` on all but the last token (decoder/model.go:1611-1626). area A
A-P02 records the cost: K=32 sequential 104.9 tok/s = ~305 ms; batched 263.4 tok/s needs a fidelity gate that was never run at
K<64. A reuse turn of 8-63 new tokens at startPos>=64 takes the batched pass, whose floor is ~96 ms at C=16
(chunked-prefill-2026-09-27.md:102-105 [rec via area A]).

What the repo already has. `metalResident.PrefillLastNArgmax` (metal/batch.go:772-803) runs consecutive positions of the bound slot
through `forwardMultiInto` in pieces of 8. Its identity gate is 0 logits and 0 K/V elements differing from production
decode in 7 cases on both models (metal-spec-step-verify:21-27 [rec], `TestMC3Verify_sameSlotRowsBitIdentical`). It is shipped
as the n-gram verify only.

Cost, counted from recorded ratios. An 8-row step on the 1.5B at depth 128 costs 1 + 7 x 0.108 = 1.76 tokens
(metal-spec-step-verify:35-41 [rec]); one token is 10.46 ms GPU (area A, citing concurrency-mc3-s0 §5 [rec]). So a piece is
~18.4 ms, 2.3 ms/token against 9.5 ms/token sequential [rec: 305/32]. K=32 is 4 pieces, ~74 ms; K=16 ~37 ms against 167;
K=8 ~18 ms against ~84. Non-final pieces do not need the LM head (the step's lm_fb4 is 1.52 ms of ~17-18 ms, S4 [rec]) and
`forwardMultiInto` has no mode that skips it; adding one is part of the change and is worth ~8% per non-final piece [cnt].
7B: a production token is 33.2 ms (concurrency-mc3-7b-w7:105-112 [rec]: the step kernel's B=1 time, 64.37 ms, is 1.94x a token); a piece is
(1 + 7 x 0.157) x 33.2 = ~70 ms, which matches the recorded B=8 step at depth 128 (68.51 ms, same table); K=32 is ~275 ms against ~1.06 s [proj]. At
depth 2048 the piece costs 2.35x (1.5B) or 2.65x (7B) a token, still under 40% of 8 sequential tokens.
Against the batched pass: ~2.3 ms/token (step) vs ~6 ms/token at C=16 [rec: 95.8 ms/16], and vs ~2.5 ms/token at K=64
(159 ms, pre-R16 [rec via area A]). The step route wins below roughly K=40 on the 1.5B and is level at K=64.

MLX analog. `QuantizedMatmul` runs qmv (one TG row per vector) below `get_qmv_batch_limit` and qmm above it; on the M1 Pro
the limit is 14/10/6 by shape (quantized.cpp:89-130, dispatch at 1890-1893 [mlx]). The 8-row step sits in MLX's qmv range, so
MLX's own crossover agrees that M<=8 should not take a GEMM tile.

Fidelity. Bit-identical to sequential decode by the record above; no §3.2 pooled gate applies. It also fixes a quirk:
`mc3Prefill` cuts chunks so the final pass has >= `prefillTailMin` tokens because a sub-8 tail "goes down the sequential path,
whose numerics are not the batched prefill's" (decoder/model.go:1666-1668); a step-route tail would equal the sequential numerics
as well. `--exact-prefill` users get a ~3.7x faster exact path at every K [proj].

Preconditions: `batchIneligible() == ""` (two or more slots, dense W4A8, no QK-norm, no windows, no adapter; metal/batch.go:301-333),
which holds for the default Qwen2.5-Coder 1.5B and 7B on `--kv-sessions 4`.
Band [proj]: K=32 fresh prompt TTFT 305 -> 70-85 ms; K=16 167 -> ~40 ms; reuse turns of 20-50 tokens roughly level with today's
~96 ms pass on the 1.5B, so gate the route to K<=~40 until measured. Basis: the two recorded ratios above.
Probe: a test that drives `forwardMultiInto` over a 32-token prompt in 4 pieces, compares all K/V bytes and the last row's
logits against the sequential loop (expect 0 differences), then times K=8/16/32/64 against sequential, against
`GOINFER_METAL_FAST_PREFILL_FLOOR=0`, and on the 7B. Kill: K=32 TTFT above 0.6x sequential, or slower than the batched pass at
K>=16 once A-P01 lands.
Why this is not already closed: red-october.md:274 and :392 treat 8-255-token prompts as "sequential, stays exact" and the only
proposed fix is lowering the floor behind a new fidelity cell (metal/backend.go:614-619); the step kernels shipped after (2026-09-26/27)
and no record routes prompts through them. A-P02 hands this alternative to area E.

**Status, 2026-10-01.** T1.10's byte comparison holds: `TestMC3Step_promptInRowsBitIdentical` feeds K = 8, 16, 32 and 64
tokens to the batched step in 8-row pieces from position 0, and every K/V element and every row's logits match the
sequential loop, on the default-run fixture and on the 1.5B (0 of 9,723,904 logits at K = 64). The timing half of T1.10
is night work.

**Status, 2026-10-02 (branch `metal-audit`): done below the floor; held above it.** T1.10's timing: step ÷ sequential
0.22 (1.5B) and 0.25 (7B) at every K; step ÷ pass 0.40 / 0.80 / 1.56 at K = 16 / 32 / 64. `PrefillLast` now routes a
prompt that ends below the floor through the step (bit-identical, tested through `PrefillLast` on the fixture and the
1.5B). The above-floor half is held: a short suffix there is also a chunked prefill's tail, and on the step it breaks
chunk invariance (C = 81, now in the gate). Details in the program doc's log.

#### E-P02 [P,B, Minor; Major if the probe lands] M-adjacent threadgroup layout for the 7B at B=2

metal/batch.go:605-619 and 634-639: at B<=`rowsQKV`/`rowsGU` the qkv and gate|up matmuls run production's GEMV once per row, each
dispatch streaming the whole weight. S4 chose it because the fragment (1.167 ms) loses to two GEMVs (2 x 0.435 = 0.87 ms) on the
7B gate|up and it is the one saving the S4 record ships (7B B=2 65.29 -> 58.76 ms at depth 128 [rec], S4:14-27). The weight is
~68 MB (0.435 ms at ~157 GB/s), so two serial dispatches stream it twice from DRAM unless the SLC (24 MB) catches the second pass,
which it cannot at this size.
MLX launches `grid_dims(M, (N+bn-1)/bn, B)` and indexes `x += tid.x * in_vec_size` (quantized.cpp:495-496; quantized.h:787-794 [mlx]), so
the M row-threadgroups of one output tile are adjacent in launch order and read the same weight tile while it is in cache.
goinfer's MC2 S0 negative covered register-shared M-row kernels (concurrency-mc2:80-140 [rec]); MLX itself gates that kernel
(`qmv_wide`) to gen>=15 for affine (quantized.cpp:541-544 [mlx]), consistent with the negative. The adjacent-TG layout with an
unchanged production body was not tried: this is not a re-proposal of the negative.
Counted ceiling: gate|up 7B at B=2 is 0.87 ms against a 0.435 ms DRAM floor; saving up to 0.43 ms x 28 = 12 ms of 58.76 ms (20%),
plus qkv (small, ~1.7 ms). Band [proj]: 0-15% of the 7B B=2 step (half the ceiling at best; 0 if the second TG misses the cache).
A 1.5B gain is unlikely (its gate|up weights, ~25 MB, are near SLC size).
Derivation: same `extractKernel`/`edit` method as batch_rows.go; tile index = tg/B, row = tg%B; bit-identical by construction
(production body, production tile order within a row). Probe: a standalone 2-row gate|up dispatch on the 7B shape; kill above
~0.65 ms. Unverifiable statically: whether concurrent TGs of one tile hit in L2/SLC on M1 Pro.

#### E-P03 [P, Minor] FB=2 shipped where S0 recorded FB=4 best on the 7B

**Status, 2026-10-03: killed and reverted.** The build's own calibration chose FB = 2 for every 7B projection
(gate|up 1.158 against 1.165 ms at FB = 4, down 0.594 against 0.584: under its 3% bar), so S0's FB = 4 advantage did
not reproduce in the production step. Graded arms were one build, 0.999–1.001×.

**Status, 2026-10-02 (branch `metal-audit`): built, bit-identical, pending its night grade.** FB = 4 is instantiated
for `mc3_bt` and `mc3_btd`, and `calibrateFB` picks per projection and per model at build. FB = 4 is taken only where the
rows divide by 32 and it is at least 3% faster. `TestMC3Step_fb4BitIdentical` forces FB = 4 everywhere: 0 logits
differ. On the 1.5B the calibration keeps FB = 2, as S0 found. The 7B's grade and its rule are in the task doc ("E-P03:
built, pending its grade").

metal/batch.go:140 and 204 instantiate `mc3_bt_fb2`, `mc3_btd_fb2`; 246 instantiates `mc3_lm_fb4`; metal/batch.go:366 binds those. S0's table
(concurrency-mc3-s0:92-103 [rec]) lists the best 8-row variant per shape: 1.5B gate/up 0.244 ms (bt fb2), 1.5B down 0.138 (btd fb2),
7B gate/up 1.167 (bt fb4), 7B down 0.579 (btd fb4), 7B qkv 0.155 and o 0.115 (fb2). The fb2 times for the two 7B shapes are not
recorded, and FB=4 exists only in test code (metal/gemm_mma8_mc3_test.go:223-224,294-295,343-354). Those two shapes cost
(1.167 + 0.579) x 28 = 48.9 ms [cnt] of the 7B's ~59-66 ms step. If fb2 is 10-12% slower there, the saving is ~5-6 ms (7-9%).
Band [proj]: 0-8% of the 7B step at B>=3. FB only changes which rows a simdgroup owns, so bit-identity is unchanged (S0 table is
the "bit-identical" column). A per-model pick at `buildBatch`, like `calibrateRows` (metal/batch.go:405-453), keeps the 1.5B on fb2.
Probe: instantiate fb4 for bt/btd, run `TestMC3Step_bitIdentical` and `TestMC3Step_throughput` on the 7B. Kill: <2% on the step.
Where the fragment's remaining cost sits [cnt]: 1.5B gate|up 0.244 ms against an MMA floor of ~0.085 ms; 7B gate|up 1.167 against
~0.417 ms (~35% of MMA peak). The rest is nibble dequant (~32 scalar ALU ops per lane per (group, block): four MMAs per 32-k group
per 8-row block), loads and accumulate, in a split the repo did not record. MLX's dequant trick (magic-number nibble to float)
would save about 2 of ~38 ops per (g, b); not worth a probe by itself.

#### E-P04 [P, Minor; Major if top_p clients count as a default shape] No device top-K for filtered sampling

Temperature-only requests sample on-device (Gumbel, R7b, 0.96-0.974x greedy [rec]). Any filter takes the host path: there is
no Metal `ResidentTopK`/`ForwardTopK` (grep: only decoder/model.go and residency.go carry the names). red-october.md:90 (Metal
column) counts "host softmax/select over 152k per token = 1.8 ms of a 13.5 ms token (counted) -> ~10-15%, unmeasured";
benchmarks.md:1104 says no Mac sampled cell existed except temp-only. optFwd is off under MC3 (metal/model.go:2091), so a batched
filtered request has no overlap either. decoder/model.go:1971-1980 records the CUDA top-p regression and its fix; Metal has no
analog. Counted 1.8/13.5 = 13% [cnt]; worse on the 0.5B.
MLX analog: none fused; mlx-lm top-p uses argsort (sort.cpp). The design to port is goinfer's own CUDA K-best plus host filter
with the `Full()` fallback that keeps identity with the host draw.
Why not closed: red-october.md:90 leaves it "unmeasured"; N-28 (Sep 12) closed only the temperature-only case via R7b.
Probe: one filtered sampled served cell (top_p 0.9, T 0.7) on the 1.5B against the same cell at top_p 1.0. Kill: gap < 3%.
Fidelity: the CUDA design keeps the host draw bit-identical when K covers the mass; the fallback is the existing host path.

#### E-P05 [P,B, Minor] Deep rows in the batched step: per-row FA pairs, no timing above 1536

metal/batch.go:621-637: a row at `attnFADepthFloor` (1536) or deeper runs `attention_fa_blk` plus `attention_fa_combine` as its own two
dispatches, in a serial encoder; rows below run `mc3_attention_rows` as one dispatch. At B=8 that is 16 attention dispatches per layer
against one, 840 dispatches per step against 420 [cnt]. At ~1.3 us per dispatch (S3 [rec]) the dispatch count is only ~0.55 ms
[cnt]. The larger cost is occupancy: one row's FA grid is nKV x S = 2 x 16 = 32 threadgroups of 128 threads on a 16-core GPU, so
each of the 8 rows runs alone, in series.
Evidence from the record: the extra cost of a row at depth 2048 is 0.193 tokens (1.5B) and 0.235 (7B), against 0.108 and 0.157 at depth
128 (metal-spec-step-verify:35-41 [rec]), a depth term of 0.085 x 10.46 = 0.89 ms per row [cnt]; the same shape for rows of different
slots. At depth 512, B=8, attention is ~7 ms of 23.36 ms (30%) [cnt by subtraction: S3 reads +4.64 ms for +384 keys at B=8;
attention 4.0 ms of 20.24 at B=4], ~8x over the DRAM floor (8 rows x 384 keys x 28 layers x 1 KB = 86 MB = 0.55 ms at 157 GB/s [cnt]);
part of that gap is the per-head kernel re-reading each K/V stripe once per Q head (6x for G=6), which the GQA-shared FA kernels avoid.
No batched-step timing exists above 1536: records show bit-identity at depths 1936/1530/100/1600 only (S3, spec-step-verify gate 1).
MLX analog. `sdpa_vector_2pass` (routed for kL>=1024 on 's'/'d' devices, scaled_dot_product_attention.cpp:~1560-1566 [mlx]) runs a
(32, gqa, qL) threadgroup per (kv head, block), so every (gqa, q-row) simdgroup of a kv head sits in one TG and reads one K/V
stripe (sdpa_vector.h:180+, 326 [mlx]); its q_len<=8 limit matches the step's 8 rows. MLX takes one key count per dispatch and
masks, so goinfer's `nKeys_rows` is the more general design.
Lever. Derive `mc3_attention_fa_rows` and `mc3_attention_fa_combine_rows` with the `edit` method (row index from the TG position,
`slotOff`, `nKeys_rows`, partial buffer indexed by row): one dispatch pair per layer, 256 TGs at B=8, bit-identical to the per-row
kernels (same body, same split S=16, which is fixed above the floor because the nKeys/32 cap never binds, metal/model.go:2889-2899).
For same-slot verify rows, the rows could additionally share the K/V stripe as in MLX's 2-pass (that part changes the reduction
shape and needs its own identity check; it is not required for the first change).
Band [proj]: 5-12% of an 8-row step at depth 2048; ceiling 21-25% (7 x (0.235 - 0.157)/2.65 = 21% on the 7B, 7 x (0.193 - 0.108)/2.35 =
25% on the 1.5B [cnt], attention-related only). Most chat depths in the records are below 1536, so this is Minor on shipped defaults.
Probe: `TestMC3StepBreakdown` at depth 2048 and 4096, B=8; a stub arm that skips FA for rows 1..7 gives the ceiling. Kill: ceiling
under 5% of the step.
REVISIT (M-16, audit-metal-2026-09-12.md:891-915, closed NEGATIVE): premise "the decode graph is nearly a chain" was measured on solo
decode. In the batched step the per-row FA dispatches are independent siblings, so the premise does not hold there. aikit does not
expose `computeCommandEncoderWithDispatchType:`, so concurrent dispatch is not a cheap probe; the multi-row kernel above reaches the
same occupancy without it.

#### E-P06 [P, Minor] One verify cost curve for every model

**Status, 2026-10-02 (branch `metal-audit`): built, lossless, pending its night grade.** A model loaded with
`Options.SpecNgram` (`--spec ngram`) measures its own curve at load, at depth 2048 (`calibrateVerifyCost`). Real
1.5B at that depth: 1.64 / 1.93 / 2.44 at 2 / 4 / 8 rows against the constant's 1.79 / 2.11 / 2.65. That is 8–9% below
it, not 12–51%: the gap below compared depth 128 against 2048 as well as model against model. The grade and its rule
are in the task doc ("E-P06: built, pending its grade").

`stepVerifyCost = {0, 1, 1.79, 1.95, 2.11, 2.25, 2.38, 2.52, 2.65, 3.65}` is the 7B at depth 2048, the dearest cell (metal/batch.go:749-752).
The 1.5B at depth 128 reads 1.59 / 1.71 / 1.76 at 2 / 4 / 8 rows (same comment), so the curve overstates it by 12% / 23% / 51%. The
controller drafts when E(d)/Cost[d+1] > 1 with E(d) = sum of alpha^i (spec-step-verify:51-58 [rec]); at d=1 that needs alpha > 0.79
on the shipped curve against 0.59 on the true 1.5B curve, so for alpha between ~0.6 and ~0.8 it declines drafts that would pay
(+7% on those rounds at alpha = 0.7: 1.70/1.59). d=8 is never chosen: Cost[9] = 3.65 loses to d=7 (8/2.65 = 3.02 against 9/3.65 = 2.47).
Band [proj]: 0-5% of 1.5B chat tok/s (chat measured 1.068x, copy 2.082x). Lossless either way. Fix: measure the curve at load once, as
`calibrateRows` does for the GEMV choice, or index by model size. Probe: `TestMC3Verify_rowCost` on the target model feeding
`AdaptiveDepth.Cost`; compare chat tok/s.

#### E-P07 [P, Minor] Qwen3-family dense models are outside MC3

**Status, 2026-10-04: built, on.** `mc3_qk_norm_rows` (derived from `qk_norm`) puts per-head QK-norm in the batched step; the identity suite reads 0 differing values on the real Qwen3-0.6B, and a qwen3 fixture pins it default-run. Its served confirmation is pre-registered (docs/tasks/task-metal-audit-2026-10.md, "E-P07").

`batchIneligible` returns "a family variant the batched step does not reproduce" for `r.qkNorm` (metal/batch.go:311-312). Qwen3 dense is a
supported family (benchmarks.md:508 lists qwen3-1.7b). It gets no batched step, no step-kernel verify (n-gram verify stays at the old
0.71-0.96 per extra row, spec-vs-batching-metal-2026-09-27) and no E-P01 route. The missing piece is small: `qk_norm`
(metal/kernels.go:2010-2025) is one TG per head on the fused qkv buffer, so a rows form is `qkv_rows + mc3_row*rowStride` by the same
`edit` method, one more dispatch per layer between the qkv matmul and `rope_rows`. Band: coverage, not a speed figure; the recorded Qwen2.5 gains it would
forgo are 1.785x 4-client aggregate on the 7B (concurrency-mc3-7b-w7:74 [rec]) and a 1.998x aggregate at B=4, depth 128 (same record, :105-112).
Probe: derive the kernel and run the same-slot and distinct-slot identity tests on qwen3-1.7b.

#### E-P08 [P, Minor; REVISIT "fused argmax"] Per-row logits copy and host argmax in the batched step

For greedy rows without a device draw, `forwardMultiInto` copies the 607,744 B row (151,936 x 4) and applies softcap/scale on the host
(metal/batch.go:707-720); `argmaxOnly` rows are argmaxed on the host after the command buffer (metal/batch.go:691-696). S4's non-GPU slope is
~0.067 ms per row (1.26/1.39/1.66 ms at B=2/4/8 [rec]), so the ceiling is 0.067 x 8 / 18.7 = 2.9% at B=8 and 1.4% at B=4 [cnt].
The recorded negative ("fused argmax") priced one token with a zero-copy view; it did not price MC3's per-row copied rows. The machinery
exists (`greedyDraw` and the gumbel two-stage reduction already run per row on `logitsB`; `Temperature=+Inf` takes the argmax path at
metal/batch.go:658-666). Output is unchanged (first-max ties). Not recommended before E-P01/E-P05; probe: route greedy rows through the existing
draw path and compare ids and step time.

#### E-G01 [G, Minor; Major if CI does not set the env] The bit-identity contract has no default-run gate

Identity is the contract of this default-on path. The tests that check it (`gemm_mma8_mc3_test.go`, `gemm_smallm_mc3_test.go`,
`mc3_concurrent_test.go`, `mc3_step_test.go`, `mc5_chunk_test.go`, `simdsum_tree_mc3_test.go`, `spec_multiturn_test.go`,
`spec_verify_identity_test.go`) skip unless `GOINFER_METAL_MC3=1` and need a real checkpoint; `spec_verify_curve`, `spec_prefill_regression`,
the gumbel tests (except the Philox KAT) and `sampled_gumbel_identity` need `GOINFER_HEAVY_TESTS`. What runs by default is the init-time
`edit` panics (metal/batch_rows.go:51-59), the `mc3_sstree` build-time check (metal/batch.go:457), `forwardn_batch_test`, parts of `kv_slots_test`, and
the LoRA tests. There is no decoder-level MC3 test in the snapshot, and no CI configuration. If CI does not export the env on a Mac runner, a
change to a shared kernel can break MC3 identity with no failing test.
Coverage gaps even when run: chunk-invariance at B>4 on distinct slots, C=512, unaligned startPos, the 7B; the kvI8 end-to-end gate is a 0.99
cosine on a tiny model (metal/kv_i8_test.go:327) and the kernel gate cos >= 0.9999, maxAbs <= 1e-3 (:197).
Probe: read the CI config; run `go test ./metal` on a Mac without the env and count skips.

**Status, 2026-10-01.** The probe: CI's metal job builds, vets (plain and tagged) and runs one device-free test, so no
Metal identity gate runs in CI, and `cmd/gate` never sets `GOINFER_METAL_MC3`. A plain `go test ./metal/` gave 209 pass
and 51 skip. Fixed for the identity subset: with the variable unset, a generated fixture (`metal/mc3_fixture_test.go`:
Qwen2.5-1.5B's attention geometry, everything else small, int4) now backs `TestMC3Step_bitIdentical`, `_bitIdenticalDeep`,
`_rowsPathBitIdentical`, `_drawsMatchForwardSample`, `TestMC3Verify_sameSlotRowsBitIdentical`,
`TestMC5_prefillChunkInvariance` and `TestSpecVerify_forwardNMatchesForward`; the last two are no longer behind
`goinfer_testhooks`. With the variable set they run on the real checkpoint as before. The suite now gives 223 pass and
41 skip, in 142 s. On the fixture, a one-ulp change to the batched GEMV's group scale fails both step checks, the steel
kernel's causal limit moved one key fails chunk invariance, and `ForwardBatch` attending one key short fails spec
verify. Still opt-in: the timings, the GEMM probes (0.5–1 GB of buffers), `TestMC3SimdSumTree` (the same check runs at
every batched-step build), and the concurrent and multi-turn tests, which generate on a real checkpoint with its
tokenizer. Of the gaps above, C=512 and chunk starts off the 32-row tiles are now covered; B > 4 on distinct slots and
the 7B are not.

#### E-C01 [C, Minor, latent] Solo-path logits alias

`metalResident.Forward` returns the shared `r.logitsHost` (metal/backend.go:557-559, "reused across calls"), filled by `finalizeLogits` (metal/model.go:1955-1969,
a 608 KB copy). In the MC3 solo path the generation consumes it after the resident section ends: `LogitProcessor` and `SampleWithInfo`
(decoder/model.go:2174-2182). Batched rows are copied (metal/batch.go:707-708) and the top-K path copies (`topKFullBuf`). A second generation's solo step
overwrites `logitsHost` at its own `finalizeLogits`, about one token of GPU time later. It is safe by timing margin only: a slow
`LogitProcessor` (grammar masks) or a descheduled goroutine could read a torn row. This is the failure the audit's N-27 declined for on CUDA
(decoder/spec_optfwd.go:202-204). Probe: two concurrent MC3 generations, one with a 30 ms-sleep `LogitProcessor`, compared with the same request alone.
Fix if confirmed: copy at the seam under MC3 only.

**Status, 2026-10-02 (branch `metal-audit`): confirmed and fixed.** `TestMC3_soloLogitsSurviveAnotherGeneration`
(default-run, the MC3 fixture on Metal, two generations: A with a 30 ms `LogitProcessor`, B with an 8 ms one) found
A's greedy tokens changed beside B on 3 of 3 runs before the fix, from the first token. Two seams, not one: the
prompt's seed, which `mc3Prefill` hands out after its exclusive section, and a solo decode token's logits. Each is now
copied while the generation still holds the resident (MC3 only; the non-MC3 path has no other writer). With either copy
removed the test fails 2 of 2 or 3 of 3; with both it passes 3 of 3. Bit-identical: the forward goldens pass (39/24/0).

#### E-C02 [C, Minor, default-off] `fp contract(fast)` restore and `GOINFER_PRECISE_MATH`

`gumbelMSLKernels` is the last block of `allKernels` and is bracketed by `#pragma METAL fp contract(off)` (metal/gumbel.go:49) and
`#pragma METAL fp contract(fast)` (:164). `mc3RowsKernels` is appended after it (metal/model.go:850) and the whole library is compiled with
`CompileLibraryPrecise` under `GOINFER_PRECISE_MATH` (metal/model.go:846-849). The rows kernels derive from kernels that compiled before the pragma, under the
build default; after it, contraction is forced to `fast`. Under the default build the two agree. Under precise math they may not, which would break
the "bodies unchanged, byte for byte" premise of metal/batch_rows.go:10-20 for that mode, and no test runs MC3 under precise math. Whether the Metal
compiler's precise mode defaults contraction to off is not settleable statically. Probe: `GOINFER_PRECISE_MATH=1` with `TestMC3Step_bitIdentical`.

#### E-P09 [P, Minor; possibly Major on the 7B] Default KV slots

`--kv-sessions` defaults to 4 (internal/serveapp/main.go:431), the resident context to 4096 (decoder/fitplan.go:184). A slot is ~117 MB on the 1.5B
(concurrency-mc1:24 [rec]); on the 7B it is 28 x 4096 x 512 x 2 x 2 B = 235 MB [cnt], so three extra slots are ~705 MB. `metalKVSlots` clamps the
count to the memory guard's budget and prints a banner when it does (metal/backend.go:377-417); MC1's clamp path was never exercised
(concurrency-mc1:58 [rec]). Whether untouched pages of a freshly allocated shared buffer count against resident memory is not settleable
statically; if Metal zero-fills them they do. Probe: resident set after load on the 7B with `--kv-sessions 1` and 4.

**Status, 2026-10-01.** Measured by T1.6's probe (`TestMC3_kvSlotFootprintProbe`) on the 1.5B rather than the 7B, since
the mechanism does not depend on the model and the 7B leaves little headroom on this Mac. Each arm in its own process,
ctx 4096, 112 MB of KV per slot: right after the build, 4 slots and 1 slot cost the same (footprint 2809 MB against
2805 and 2811, IOAccelerator 929 against 927), so untouched pages do not count at allocation. After one token they
all do: IOAccelerator rises by 449 MB with 4 slots and by 114 MB with 1, though only slot 0 was written, consistent with
a command buffer making each layer's one KV allocation resident. The 3 extra slots therefore cost about 335 MB from the
first token on; on the 7B, about 705 MB by the same bytes. Log: `docs/measurements/metal-audit-2026-10/t16-kv-slot-footprint-2026-10-01.log`.
Whether to lower the default slot count is the owner's call (the fix above).

**Status, 2026-10-02 (branch `metal-audit`): done, on the owner's go ("a Metal-only default of 2 KV slots").** With
`-kv-sessions` not given, serve marks the slot count as its default (`decoder.Options.ResidentKVSlotsDefault`) and Metal
keeps `metalDefaultKVSlots` = 2 (`metalKVSlots`), printing that it did. A given `-kv-sessions N` keeps N, the memory guard
permitting. The CPU session count, CUDA and WebGPU are unchanged. That saves about 224 MB on the 1.5B and 470 MB on the 7B
from the first token. Two slots keep MC3's batched step and a second conversation's prefix; a third and fourth concurrent
client wait for a slot. `TestMetalKVSlots_default` and `TestKVSessions_defaultReachesTheBackend` pin both halves, and
each fails with its half removed. The banner's slot line names the default instead of blaming the memory guard. The
graded MC3 numbers at 4 clients were measured with 4 slots, which now takes `-kv-sessions 4`.

#### E-P10 [P, Minor, opt-in] `kv_store_i8`

`e.Dispatch(r.pKvI8, g.nKV, 1, ...)` (metal/model.go:3010-3011) launches nKV one-thread threadgroups, each running the serial loop at metal/kernels.go:907-929.
28 dispatches x (a few us) ~ 0.15 ms of ~13.5 ms ~ 1% [cnt, unmeasured]. `--kv i8` is one slot, no FA, no batching (metal/batch.go:311;
metal/model.go:1309-1321, `allocSlots` stays 1 for int8 KV), so it forfeits MC1, MC3 and the step-kernel verify. Noted so the cost of the flag is on the page; no change proposed.

#### E-N01 [N, Minor] Pack dispatches

4 `pack` dispatches per layer (metal/batch.go:609, 639, 643, and the one after the first rms), 112 per step. Each only changes layout, so fusing it into
the producing rows kernel is bit-identical. At ~1.3 us each (S3) that is ~0.15 ms = 0.85% of the 17.2 ms step [cnt]. Not worth running first.

#### E-D01 [D, Minor] Stale text

`--spec` help ends "Wins ... on the CPU backend" (internal/serveapp/main.go:453); the Metal step-kernel verify shipped 2026-09-27 (2.08x on copy
traffic, 1.07x on chat, 1.5B). `Options.EmbedInt4`'s comment says "default off" (decoder/model.go:382-388) against the owner decision recorded at
task-never-swap-2026-09.md:281-283. `attnFACoreCount = 14` (metal/model.go:2698) is dead on the shipped path because `attnFABlkSplit > 0` overrides it
(metal/model.go:2889-2899); area B owns that.

### (d) Carry-forward

| ID | Status | Evidence |
|---|---|---|
| M-08 (`lora_delta` one threadgroup) | CLOSED-VERIFIED (code); still unmeasured on a Mac | metal/kernels.go:1971-2010 now launches ceil(Out/256) threadgroups, each recomputing the per-rank tree reduction (redundant, kept fused by decision, audit:1756-1757); metal/lora.go:200-229; lora_delta_multitg_test.go. No Mac adapter-decode record in docs/measurements. |
| C-03 (adapter dims unchecked) | CLOSED-VERIFIED | decoder/lora.go:171-202 `validateComputeTimeDims`, called at ~377-384 before the resident bind |
| C-04 (SetAdapter error path leaks) | CLOSED-VERIFIED | lora.go:~166-209: stopExec first, deferred `releaseLoRALayers` on partial failure; lora_bind_leak_test.go |
| N-26 (ForwardN per-token sync) | CLOSED-VERIFIED | decoder/residency.go:62-68; metal/backend.go:933-940, ForwardBatch layer-major since a1640a6a; the R12 kill is recorded |
| N-27 (zero-copy logits) | NEGATIVE-CLOSED, unchanged | metal/model.go:1955-1969 still copies; declined for the CUDA DMA-race reason (audit:1466-1480). E-C01 is the one place the same hazard is latent. |
| N-28 (sampler idle gap above T 0.2) | PARTIAL | temperature-only is on-device since R7b (gumbel.go, gumbel_sample.go); filtered sampling is still host-side (E-P04) |
| N-03 (no ResidentGreedy on Metal) | still holds | `metalResident` has no `ResidentGreedy`/`ForwardArgmax`; `ForwardArgmax` (metal/model.go:2205-2252) is test-only. The n-gram loop uses `PrefillLastNArgmax` instead. |
| batched small-M verify | superseded: shipped | step-kernel verify, metal-spec-step-verify-2026-09-27.md; the old Theta ~1.02 premise no longer applies |
| fused argmax | recorded negative; REVISIT for MC3 rows only | E-P08, ceiling 2.9% at B=8 |
| M-16 (concurrent dispatch) | NEGATIVE-CLOSED; REVISIT for the batched step | E-P05 |
| N-10 (ForwardArgmax comment) | still accurate | metal/model.go:2205 |

### (e) Checked and found correct

- Slot ordering. Per layer, `kv_store_rows` precedes attention in one serial encoder (metal/batch.go:621-637); rows on one slot are accepted only as one
  sequence's consecutive positions (metal/batch.go:530-539), so each row sees the rows below it as sequential decode does. Two sequences on one slot are refused.
- Stale KV on slot reuse. Attention is bounded by each row's `nKeys` (metal/batch_rows.go:146-149) and the FA rows by their own `uNKeys[m]`, so bytes past the
  position are never read; a reused slot needs no clearing. Spec rollback by a no-op `TruncateTo` is safe for the same reason.
- Slot addressing. `slotOffB = slot * kvSlotBytes[0] / 2` in f16 elements (metal/batch.go:568); `batchIneligible` requires uniform layers, one f16 allocation
  per layer (`kvContig`), and widths that tile, so the offset cannot overflow uint32 for the slot counts the guard allows.
- Sampled-token join. The RNG draw is taken before submit (decoder/model.go:~2152-2158), so a batched stream matches the unbatched one; error paths
  forget ids; the batched draws reuse `gumbelBKey/BIdx` serially inside one encoder (batch.go:~652-674), and the executor never races
  `ForwardSample`/`ForwardArgmax` (`stopExec` at metal/batch.go:551).
- Gumbel. Philox4x32-10, 12-term polynomial small-w branch, `precise::log`, two-stage reduction with lowest-index ties; the pragma bracket is correct under the
  default build; recorded 0/15,840 key mismatches and 12,000/12,000 identical tokens [rec]. The contract is statistical-near-exact against the host f64
  reference (<= 1e-4 mismatch, near-ties) and deterministic given logits bits and (seed, draw). `SampleAvailable` declines for softcap, logit scale and paged MoE.
- n-gram verify: `finishTrailing` fixes the one-token-short cache; `specRoundDraftWidth` clamps to the cap; ties break first-max in both argmax helpers;
  a one-row piece is production's own `Forward`.
- LoRA: `SetAdapter` stops the executor first, caches by data pointer, bounds rank at 256; batched step and the FA plan reject `loraLayers != nil`; the MC3 claim
  requires `lora == nil` (decoder/model.go:~1766).
- MC5: chunk-invariance record 0 of 14.3M KV elements differ (chunked-prefill-2026-09-27); the 413-guard fix is in.
- Executor: encode-ahead is parked on its headroom (S4:31-44 [rec], non-GPU ms/step 1.26/1.39/1.66 at B=2/4/8); `stopExec` costs only at step-to-solo
  transitions.

### (f) Not settleable statically

1. (Resolved at consolidation: E-X01.)
2. Whether CI sets `GOINFER_METAL_MC3` / `GOINFER_HEAVY_TESTS` (E-G01); the snapshot has no CI config and no decoder tests.
3. fb2 versus fb4 timings on the 7B, and the dequant versus MMA split inside the fragment (E-P03): the S0 logs are not in the snapshot.
4. Whether adjacent same-tile threadgroups share weight reads through L2/SLC on M1 Pro (E-P02).
5. Batched-step cost above `attnFADepthFloor`, and whether a multi-row FA dispatch fills the GPU (E-P05).
6. Whether untouched pages of the extra KV slots count against resident memory (E-P09).
7. Whether Metal's precise-math mode defaults `fp contract` to off (E-C02).
8. `simd_sum` tree on M3 and later: `batchSimdSumTreeOK` guards it at build, but only an M3+ run shows it holding.
9. Stability of `calibrateRows` (metal/batch.go:405-453) run to run; its choice at B=2/3 on the 7B decides how much E-P02 can buy.
10. `docs/tasks/task-concurrency-2026-09.md` and its log directories are cited by many records and absent from the snapshot.


## 9. Area F — Correctness of recently changed code, gates and tests, stale claims

Reviewer F. Static review of goinfer HEAD 844700f8, aikit v1.51.0 and MLX 9c3d355. Nothing was built or run. Counts are from `grep`/`wc` over the snapshot; every figure that is not a count cites the doc line it comes from. The snapshot has no `testdata/`, `scripts/`, `cmd/`, `internal/prequant`, `decoder/*_test.go`, `docs/completed/`, `docs/hardware-matrix.md`, or most of `docs/tasks/`; items that depend on those are listed in section (f).

### (a) Summary

1. Two findings are wrong-output or memory-corruption bugs on shipped options, not on the default configuration: `PrefillLast` has no int8-KV guard (F-C01), and the exact prefill kernel indexes a 4096-float threadgroup array by absolute key position (F-C02). F-C02 becomes reachable on an unpinned load whenever the fit guard auto-pins a context between 4097 and 32768, which the repo's own records show happens (5689, 12109-29666 positions).
2. R19's `attention_prefill_steel` is the production prefill attention for every head-dim-128 model and has no assertion in any test that runs by default. The only comparison against another kernel is log-only inside an env-gated harness. The G-08 closure test uses head dim 16, so it exercises the fused kernel, not steel, and nothing exercises steel at `startPos > 0`, which is what chunked prefill and every prefix-reuse turn do (F-G01).
3. The MC3 batched step, chunked prefill (default `-prefill-chunk 512` under MC3) and spec-verify tests only run with `GOINFER_METAL_MC3=1` plus a real checkpoint. The chunk-invariance record was taken at 04:43-05:03 PDT on 2026-09-27, about 13 hours before the steel kernel's gate run (F-G02).
4. `TestMetalCtxCapWithinKernelBound` asserts two constants and is documented as what keeps the 32768 ceiling "a FACT"; no prefill kernel is involved (F-G03).
5. Claims: `docs/benchmarks.md` still carries pre-R19 and pre-R18b statements next to the newer rows (F-D01); `alias.go` says every S6 gate passed where the S6 record says one was not met literally and one was not run on the shape it targets (F-D02); `HiddenLast`'s doc says "bit-identical by construction" while its own tests measure cosine 0.9991-0.9993 (F-D03).
6. R17's `attention_fa_blk` G=6,7 path, R18/R18b GEMV tiling, the f16-scale plumbing (every reader is `half`, one conversion function), the S6 alias lifetime/alignment/Close ordering, MC1 slot bookkeeping, Gemma 1/CodeGemma admission and the Gemma 2 decline all hold up on reading (section e).
7. Of the old audit's entries: C-01..C-06 are all CLOSED-VERIFIED in code; G-01, G-06, G-07 CLOSED-VERIFIED; G-02, G-03, G-04, G-08 PARTIAL; G-05, G-09 OPEN; N-items mostly CLOSED-VERIFIED, with N-15, N-16, N-19, N-25, N-40, N-41 still open (section d).
8. Test census (counts, `metal/*_test.go`): 200 files, 293 top-level `Test` funcs. 98 files (154 tests) are `//go:build darwin` and run with a plain `go test ./metal/`. 102 files (139 tests) need `-tags goinfer_testhooks`. 67 of the 98 default files and 82 of the 102 hooks files contain at least one `t.Skip`; 59 files reference `GOINFER_HEAVY_TESTS`/`requireHeavyModel`. The S6 record's full hooks run on the author's Mac was 171 pass / 74 skip / 0 fail (`s6-alias-2026-09-24.md:233`), so about 30% of the hooks suite skipped even there.

### (b) Findings table, ordered by effect

Severity qualifier "(non-default option)" means the condition is an explicit flag or a fit-guard result, not the shipped default configuration (f16 KV, 4096 context, fused or steel attention). The brief's Critical definition says "default path"; I kept the three-level scale and flagged the condition so you can reclassify.

| ID | Sev | Claim | Evidence | MLX analog | Band | Probe |
|---|---|---|---|---|---|---|
| F-C01 | Critical (non-default option: `--kv i8`) | `PrefillLast` never checks `r.kvI8`; `kv_store_f16` writes half K/V into int8-sized KV buffers (wrong KV for every prompt token, OOB device write once position >= ctxCap/2) | `metal/backend.go:763-843`, `metal/model.go:1027-1029`, `:1145-1163`, `metal/prefill.go:676-683,1270`; decode/UploadKV/batch do check `kvI8` (`metal/backend.go:1023`, `metal/batch.go:311`) | MLX has one KV dtype per cache; no analog | correctness | llama-tiny, `KVPrecision:"i8"`, PrefillLast(16 embs, floor 0) vs 16x Forward |
| F-C02 | Critical (non-default option: ctx > 4096 and head dim > 128, or `GOINFER_METAL_FUSED_ATTENTION=0`) | exact `attention_prefill` stores scores at `sc[s]` with absolute `s` into `threadgroup float sc[4096]`; nothing bounds `startPos+M` by 4096 on that path | `metal/prefill.go:363,370,382`, `:313` (comment admits it), `:1041-1085`, `metal/backend.go:817`, `metal/model.go:28-29,44-62` | `sdpa_vector.h:56-58,91-130`: online softmax, no key-length-proportional threadgroup buffer | correctness | hd=256 fixture, ctxCap 8192, M=5000 vs sequential Forward |
| F-G01 | Major | the production hd=128 prefill kernel has no default-runnable assertion; coverage is one pooled gate at `startPos=0` plus log-only A/B | `metal/prefill_attn_r19_test.go:25-144` (only `hb(...)` logging), `metal/prefill_decomp_test.go:44-45` (env gate), `metal/prefill_gate_ref_test.go:540`, `metal/prefill_startpos_test.go:40,49-50` + `metal/moe_model_test.go:25` (hd=16) | `steel_attention.h:11-16,282-299,365-412` | gate | float64-reference test as `metal/attn_fa_blk_test.go:147` does for R17 |
| F-G02 | Major | MC3 step, chunked prefill and spec-verify gates are env-gated on a real checkpoint; the chunk-invariance record predates the steel gate | `metal/mc5_chunk_test.go:22-24,86-88`; 61 `GOINFER_METAL_MC3` occurrences in 8 test files; `chunked-prefill-2026-09-27.md:15-16,64,130` vs `metal-prefill-attn-2026-09-27.md:14,91` | none | gate | re-run `TestMC5_prefillChunkInvariance` on the wired build |
| F-G03 | Major | `TestMetalCtxCapWithinKernelBound` asserts two constants; its comment claims it keeps the ceiling a fact (**fixed 2026-10-01**, the test removed) | `metal/resident_cap_test.go`; `metal/model.go:28-29` | none | gate | replace with a prefill run at nKeys = 4097 on hd=256 |
| F-C03 | Minor (latent) | `attention_fa` has `ATTN_FA_MAXG 8` arrays and no G <= 8 guard on the Go side; the comment says only hd=128 is enforced | `metal/kernels.go:1198,1222,1225,1261`; `metal/model.go:1266-1270,1502-1512`, `:2555-2571` | `steel` has no G array; n/a | correctness | list registry archs with hd=128 and nH/nKV > 8 |
| F-D01 | Minor | `benchmarks.md` Metal prefill row, §B3 banner and §B3 bullets contradict R19, R18b and the headline decode row | `docs/benchmarks.md:43,47,1028-1031,1080-1081` vs `metal-prefill-attn-2026-09-27.md:122,126` | none | doc | edit |
| F-D02 | Minor | `alias.go` says every S6 gate passed; the record says otherwise for gate 1 (literal reading) and the hog arm (M26 not run); v15 kinds 3/4/5 carry binary16 scales but are not aliased and the banner says they carry none | `metal/alias.go:37-40,26-28,302-306`; `s6-alias-2026-09-24.md:131,147-161,191-194,284`; `decoder/model.go:172-186`, `decoder/serialize.go:1738-1743,1810-1822,1895,1970` | none | up to ~390 MB copied on the 7B for a non-metal-target bundle (record's own figure, `s6-alias-2026-09-24.md:142`) | edit; add v15-non-metal fixture |
| F-D03 | Minor | `HiddenLast` doc says "bit-identical to the CPU reference by construction"; its own tests measure 0.9991-0.9993 / 0.99985 | `metal/backend.go:891-894`; `metal/hiddenlast_resident_parity_test.go:12-21`; `metal/prompthidden_resident_parity_test.go:17-22` | none | doc | edit |
| F-D04 | Minor | `gpu-residency-coverage.md` says Qwen-VL multimodal resident decode parity is "verified" in two tests that run on text fixtures | `docs/gpu-residency-coverage.md:134-138`; `metal/forwardmrope_parity_test.go:33-53` | none | doc | edit |
| F-G04 | Minor | `cmd/gate gpu` gives false failures on a fresh checkout (gitignored fixtures); the "decode==verify" gate it cited is not in the tree at HEAD | `c3-metal-consumer-window-v0.18.0.md:123-133,185-238` | none | gate | check where `TestBatchedVerifyKernelParity` went |
| F-N01 | Minor | `HiddenLast` on a multi-slot resident overwrites the bound slot from position 0 and forgets its ids, evicting that conversation's reuse | `decoder/embed.go:56-65,83-89`, `decoder/resident_reuse.go:391-396` | none | one full re-prefill of the evicted conversation per embeddings request | pick a free slot first |
| F-N02 | Minor | S6's changing-file safety rests on one comment (temp+rename); MAP_SHARED means an external in-place write changes live weights | `decoder/giwmap_darwin.go:11-25,43` | none | n/a | none statically |
| F-B01 | Minor | steel rounds P to half before PV; MLX keeps P in f32 and uses a mixed-type MMA | `metal/prefill.go:648-650` vs `steel_attention.h:473-521` | yes (the delta) | fidelity only | gate pooled KL with P in f32 |

### (c) Full entries

#### F-C01. `PrefillLast` writes f16 KV into an int8 KV cache (Critical, non-default option; the same defect as A-C01)

- **What.** With `--kv i8` (`decoder.Options.KVPrecision == "i8"`, `decoder/model.go:516,232`; plumbed from `internal/serveapp/main.go:186`), `buildResident` sets `r.kvI8 = m.KVCacheI8()` (`metal/model.go:929`) and allocates each layer's K and V as `paddedCtxCap*kvDim*1` bytes plus separate f32 scale buffers (`metal/model.go:1308-1312`, `byteBuf(d, kvBytes*allocSlots)` at `:1162-1163`). `PrefillLast` (`metal/backend.go:763-843`) checks: fast prefill enabled, the floor, `prefillOK`, and `startPos+len <= ctxCap`. `prefillOK` (`metal/model.go:1027-1029`) is a function of model features, geometry, MoE and DeltaNet; it does not read `kvI8`. The batched pass then dispatches `kv_store_f16` over `r.kc[l]`/`r.vc[l]` (`metal/prefill.go:1286`), a kernel that does `kc[pos*kvDim + i] = qkv[...]` on a `device half*` (`metal/prefill.go:676-683`).
- **Failure.** Every prompt of at least 64 tokens (floor, `metal/backend.go:797`) with a suffix of at least 8 tokens (`decoder/model.go:1565`, `residentPrefillSeed`) takes this path by default. The K/V rows land as f16 bit patterns in an int8-typed cache and the scale buffers are never written, so decode's `attention_i8` reads wrong K/V for the whole prompt. Once `pos*kvDim*2` bytes exceeds the buffer (position >= paddedCtxCap/2, i.e. a prompt over about 2048 tokens at the 4096 default) the write runs past the end of the MTLBuffer; `checkCap`'s own comment (`metal/backend.go:516-522`) says such writes corrupt adjacent buffers on unified memory.
- **Why it is plausible nobody saw it.** The int8-KV tests drive only sequential `Forward` and `UploadKV`: `metal/kv_i8_test.go:255-330` steps 5 tokens through `Forward`; `metal/uploadkv_parity_test.go:235` covers `UploadKV`. No test calls `PrefillLast` with `kvI8` set (`grep` for `KVPrecision` in `metal/*_test.go` returns only `metal/kv_i8_test.go:270` and `metal/residentkv_alloc_test.go:38,78`). The other batched paths exclude `kvI8` explicitly (`metal/batch.go:311,489`, `metal/model.go:2870`); prefill is the one that does not.
- **Why this is not already closed.** I checked `docs/audit-metal-2026-09-12.md` (no entry mentions kvI8 and prefill together; C-01 is about the fused kernel's tail read), `metal/backend.go:496-501` (`fastPrefill` keys on `exact` and two env knobs only), and `decoder/model.go:1554-1596` (no KV-precision condition on the `Prefiller` call).
- **Fix and fidelity.** Return an error from `PrefillLast` when `a.r.kvI8` (the caller then falls back to the sequential loop, `warnPrefillDeclined`). This changes no numerics: the sequential path is the one `--kv i8` already uses today in effect (and the one the tests cover). Writing an int8 prefill store kernel is a larger change that would need its own gate.
- **Probe.** `decoder.Load("testdata/llama-tiny", {Quant:"int4", KVPrecision:"i8", ResidentContext:64})`, `buildResident`, `setResidentKnob(..."GOINFER_METAL_FAST_PREFILL_FLOOR","0")`, `PrefillLast(16 embeddings, 0)` against 16 sequential `Forward` calls. Kill criterion: logits cosine >= 0.99 and the call returning an error both count as the guard working.
- **Not reproduced** (no device). Confidence: confirmed by reading the allocation, the dispatch and the missing guard; the outcome description is inferred from the index arithmetic.

#### F-C02. Exact `attention_prefill` overflows `sc[4096]` above 4096 keys (Critical, non-default option)

- **What.** The exact kernel (`metal/prefill.go:349-387`) declares `threadgroup float sc[4096]` (`:279`) and stores `sc[s]` for `s` in `[winStart, nKeys)` using the absolute key index (`:286`, `:298`), so any row with `nKeys > 4096` writes past the array, including windowed layers (the window only moves `winStart`). The fused kernel's comment states the overrun outright (`metal/prefill.go:396-397`: "the exact kernel above allocates sc[4096] and would silently overrun a longer context"). The steel and fused kernels are tile-online and do not have the limit.
- **When the exact kernel runs.** `useFusedAttn` needs `GOINFER_METAL_FUSED_ATTENTION` on and `hd%8==0 && hd<=128` (`metal/prefill.go:1073`); `useSteelAttn` needs `hd==128` (`:920`); otherwise `pAttn` (`:1083-1085`). So: every head-dim-256 family admitted to prefill (Gemma 1/CodeGemma after 08926838a, Gemma 3 after M-06, decided by `prefillFeatures` at `metal/model.go:103-119`), and any model with `GOINFER_METAL_FUSED_ATTENTION=0`.
- **When nKeys can exceed 4096.** `PrefillLast` bounds only `startPos+len <= ctxCap` (`metal/backend.go:817`). `ctxCap` is 4096 for an unpinned load but up to 32768 for an explicit `-ctx`, or for an unpinned load the fit guard auto-pins (`decoder/model.go:702-709` writes the guard's shrunk context into `opts.ResidentContext`; `resolveMetalCtxCap`, `metal/model.go:44-62`, then honours anything up to 32768). The repo records the guard picking values in that range: 5689 positions (`c3-metal-consumer-window-v0.18.0.md:140`) and 12109-29666 (old audit N-41, `audit-metal-2026-09-12.md:1597`), measured under memory pressure on the Mac. A prompt (or a prefix-reuse turn at `startPos` plus suffix) over 4096 tokens then reaches the exact kernel on a Gemma-class model.
- **Failure.** Out-of-range threadgroup writes and reads: wrong attention rows or undefined behaviour; no error is raised.
- **Why this is not already closed.** `docs/audit-metal-2026-09-12.md` C-01 (:924-947) concerns the fused kernel's tail read and is closed by padding the allocation; no entry covers the exact kernel's score array. The comment at `metal/model.go:28-29` ("allowing deep context up to metalCtxCapMax without threadgroup memory overflow") is true of the decode kernels (`metal/kernels.go:1014,1069-1083,1579,1600,1694,1738` are tiled) and false of this one. `metal/resident_cap_test.go` (F-G03) does not touch it. `metal/attn_shape_test.go:55-77` covers decode `attention` to nKeys 32768, not prefill.
- **Fix and fidelity.** Decline to the sequential path when `!useFusedAttn` and `startPos+M > attnScoreTileBound` (one condition beside `metal/backend.go:817`); sequential decode attention is tiled and is the bit-identity reference, so nothing changes numerically. Tiling the exact kernel would change its reduction order only beyond 4096 keys, where nothing exists to be identical to.
- **MLX analog.** `sdpa_vector.h:56-58` allocates `outputs[BN*BD]`, `max_scores[BN]`, `sum_exp_scores[BN]` (32x32 plus 2x32 floats) regardless of key count and runs an online softmax (`:91-130`).
- **Probe.** An hd=256, 2-layer synthetic fixture with `ResidentContext: 8192`, `PrefillLast` of 5000 embeddings (floor knob 0) against sequential `Forward`; plus a variant with `GOINFER_METAL_FUSED_ATTENTION=0` on an hd=64 fixture. Kill: cosine >= 0.999 on both.

#### F-G01. `attention_prefill_steel` is ungated by default (Major)

- **What is covered.** `grep` for `attention_prefill_steel|pAttnSteel` across `metal/` finds only production (`prefill.go`), `prefill_attn_r19_test.go`, `prefill_decomp_test.go` and `prefill_gate_ref_test.go`. `runR19Phase` (`metal/prefill_attn_r19_test.go:25-144`) computes relative L2, max diff, cosine and timings, and reports all of them through `hb(...)`; it contains no `t.Fatal`/`t.Error`, and it runs inside `TestMetalPrefillDecomp`, which skips unless `GOINFER_METAL_DECOMP=1` and a real checkpoint exists (`metal/prefill_decomp_test.go:44-45,56`). The fidelity evidence is the §3.2 pooled gate run once, on set A, with S and D7 (`metal-prefill-attn-2026-09-27.md:88-107`): K = 256 / 512 / 1024 plus one K=3900 cell on S.
- **What that does not see.** (1) `startPos > 0`: the gate calls `PrefillLast(ctx, embs, 0)` (`metal/prefill_gate_ref_test.go:540`, old G-08). The G-08 closure test (`metal/prefill_startpos_test.go:40-98`) runs on `genTinyWeights`, whose head dim is 16 (`metal/moe_model_test.go:25`), so it dispatches the fused kernel; `useSteelAttn` needs `hd==128` (`metal/prefill.go:1074`). Under MC3 chunked prefill every chunk after the first runs steel at `startPos` 512, 1024, ...; every prefix-reuse turn does too. (2) Ragged M and K: gate cells are multiples of 32 except the one K=3900 cell (3900 mod 32 = 28, mod 16 = 12) on S. (3) Windows: the kernel's window path (`metal/prefill.go:568,574,579`) is reached at hd=128 only by a sliding-window hd=128 model, which no gate cell uses. (4) GQA groups: S has G=6 (12/2) and D7 G=7 (28/4); other groups at hd=128 (4, 8) are uncovered. (5) The baseline run on the retired fused kernel is "owed" (`metal-prefill-attn-2026-09-27.md:108-109,127`).
- **The decoder does pass `startPos > 0`.** `residentPrefillSeed` calls `PrefillLast(ctx, embs, from)` with the reused-prefix offset (`decoder/model.go:1583`), and the MC3 chunk loop calls it with `from` advancing by the chunk size (`decoder/model.go:1681`). The comment at `metal/backend.go:834-836` ("Unreachable today (the decoder always passes 0)") is therefore stale, which is a small instance of the CLAUDE.md doc-comment rule.
- **Reading of the kernel.** My read of `metal/prefill.go:533-672` found no defect (section e). The finding is that nothing would notice one.
- **Why this is not already closed.** `metal-prefill-attn-2026-09-27.md:124-127` lists "Owed" as served TTFT and the fused baseline gate; it does not list a default-runnable correctness test. `metal/prefill_attn_r19_test.go:11-14` calls the remaining harness "the comparison against the retired kernel", which is the log-only phase. CLAUDE.md's rule that a doc comment claiming coverage with no assertion naming the thing is worse than silence applies to the G-08 closure note (`audit-metal-2026-09-12.md:1226-1237`), which says the `startPos` masking is now covered.
- **Fix.** One default test in the shape of `metal/attn_fa_blk_test.go:147` (float64 reference over the same f16 K/V): hd=128; M in {1, 7, 31, 33, 100}; startPos in {0, 5, 16, 1000}; window in {0, 64}; nH/nKV in {12/2, 14/2, 32/8}; K/V buffer allocated to exactly `startPos+M` rows rounded to 8 (not page-rounded); assert cosine >= 0.9999, max abs below the fused test's 0.05 bar (`metal/attention_prefill_fused_test.go:109-112`) tightened to what the kernel measures, and no NaN. Mutation-check by shifting the causal limit by one.
- **Fidelity implication.** None (test only).
- **Status, 2026-10-01.** Done to this spec: `TestAttentionPrefillSteelMatchesFloat64` runs by default, 240 cases (the
  grid above, with uniform and peaked scores), in about 3 s. Worst row-head cosine 0.9999999, worst max abs 4.1e-4; the
  bar is 1e-3. A causal limit shifted by one key either way fails 234 and 239 of the 240 cases. The `metal/backend.go`
  comment is corrected.

#### F-G02. MC3 / chunked-prefill / spec-verify gates are env-gated, and the chunk-invariance record predates steel (Major)

- **What.** `TestMC5_prefillChunkInvariance`, `TestMC5_chunkCost`, `TestMC5_passCost` skip unless `GOINFER_METAL_MC3=1` (`metal/mc5_chunk_test.go:22-24,86-88`) and a checkpoint at `GOINFER_METAL_MC3_MODEL`. The same variable gates 61 occurrences across 8 files (`mc3_step_test.go`, `mc3_concurrent_test.go`, `spec_multiturn_test.go`, `spec_verify_identity_test.go`, `gemm_smallm_mc3_test.go`, `simdsum_tree_mc3_test.go`, `mc5_chunk_test.go`, `gemm_mma8_mc3_test.go`). All of `mc5_chunk_test.go` also needs `goinfer_testhooks`.
- **Ordering.** The chunk-invariance table (0 of 14,336,000 KV elements and 0 of 151,936 logits differ, `chunked-prefill-2026-09-27.md:15-16,35-40`) was produced in the 04:43-05:03 PDT runs (`:64,130`); the steel kernel's gate run is 18:29-19:01 PDT and its wiring is recorded afterwards (`metal-prefill-attn-2026-09-27.md:91,115-122`). The shipped default is `serve -prefill-chunk 512` (`chunked-prefill-2026-09-27.md:10`). So the record's headline "Metal prefill is chunk-invariant" describes the fused-kernel build.
- **Reading.** Steel keeps all online-softmax state per row and walks absolute-aligned 16-key blocks; a row whose block is fully masked takes the `mNew > -INFINITY` branch and changes nothing (`metal/prefill.go:639-643`), so per-row results should not depend on how rows are grouped into 32-row tiles. That is an argument from the kernel's structure, not a measurement.
- **Why this is not already closed.** The R19 record's owed list (`:124-127`) does not include re-running chunk invariance; the chunked-prefill record has no later annotation (`grep` for "steel|R19" in it returns nothing).
- **Fix.** Make the invariance test run by default on a generated hd=128 random-weight fixture (it needs random weights and two KV slots, not a real checkpoint), then re-run on the 1.5B once and append the result to the record.
- **Status, 2026-10-01.** Done. The invariance test runs by default on the fixture (E-G01's status names the rest of the
  subset), adds C = 512, 100 and 77, and was re-run on the 1.5B on the steel build: 0 of 14,336,000 K/V elements and 0 of
  151,936 logits differ at every size, and the record has a dated addendum. The aligned sizes alone could not see a bug at
  a tile edge: with the steel kernel's causal limit moved one key, all four still matched bit for bit, and only the
  control failed.
- **Fidelity implication.** None.

#### F-G03. The context-ceiling "fact" test asserts constants (Major by the brief's gate rule)

- `metal/resident_cap_test.go`: the comment says the test "pins the invariant that keeps the resident context ceiling a FACT" about the attention kernel's score tile; the body checks `metalCtxCapDefault <= attnScoreTileBound` and `metalCtxCapMax % attnScoreTileBound == 0`. It cannot detect F-C02, and `metal/model.go:28-29` repeats the claim for "the attention kernel". It runs by default (`//go:build darwin`, no device needed) and so reads as coverage.
- **Fix.** Delete the claim and add the F-C02 prefill probe, or keep the arithmetic check under a name that says what it checks.

#### F-C03. `attention_fa` has no group-size guard (Minor, latent)

- `metal/kernels.go:1198` defines `ATTN_FA_MAXG 8` and sizes per-thread arrays with it (`:1088,1091,1127`); the kernel comment says the hd=128 requirement is "enforced by the Go dispatch site (canUseAttnFA), NOT in-kernel" (`:1038-1039`) and nothing on the Go side enforces G <= 8: `canUseAttnFA` (`metal/model.go:2866-2882`) tests hd==128, nKV>0, depth, and family flags; the R17 selection (`:1333-1339`) picks `attention_fa_blk_g6/g7` for G = 6, 7 and leaves every other G on `attention_fa`. A dense hd=128 layer with nH/nKV > 8 would index past the arrays at depth >= 1536. No checkpoint that fits a 16 GB Mac that I know of has such a layer (GLM-4.5-class has 96 heads / 8 KV heads at hd=128, so G=12, with a dense first layer, but is far over the memory budget); I did not enumerate the registry.
- **Fix.** `canUseAttnFA` and `canUseAttnFAAt` return false when `nH/nKV > 8`; one comparison. No numeric change.

#### F-D01. `benchmarks.md` Metal rows disagree with each other and with the records (Minor)

- **Prefill headline row** (`docs/benchmarks.md:43`): "1.96x behind at K=3900 ... the K=3900 remainder is attention, which this change does not touch". R19 shipped attention (7.2x in sequence), and its record gives `PrefillLast` wall at K=3900 8171 -> 4316 ms, "in process 4.32 s against Ollama's served 4.23 s", explicitly "a projection, not the served claim" (`metal-prefill-attn-2026-09-27.md:122,125-126`). `grep` for "steel", "R19" and "metal-prefill-attn" finds no occurrence in `benchmarks.md` outside one unrelated long line. The row needs a dated callout pointing at the R19 record and saying the served number is owed.
- **Decode headline row** (`:47`) leads with the pre-registered sweep (behind at depth), then updates through R17, R18, R18b to "AHEAD of Ollama in every cell" on same-session interleaved A/Bs that the row itself labels "NOT the pre-registered sweep". Its own 7B @ 2048 cell is "not graded: Ollama read 20% below its own morning value". That is disclosed, but the AHEAD sentence is the headline.
- **§B3** still opens with "Superseded 2026-09-25 ... BEHIND at every 2048/3900 cell (0.58-0.75x)" (`:1028-1031`) and, in the verdict bullets, "Do not quote a Metal speed multiple as a headline ... The defensible Metal claims are portability" (`:1080-1081`). Both contradict `:47`. The §B3 banner was not updated when `:47` was.
- **N-40** (old audit) is still open: the §B3 heading says "4-bit both sides" (`:1026`) with no mention that the LM head runs int8.
- **Fix.** Callouts at `:43`, `:1028` and `:1080`; no number needs inventing.

#### F-D02. `alias.go` overstates S6's gates; the v15 scale case is not covered (Minor)

**Status, 2026-10-03: done for the common path.**
- **The header and banner:** fixed in T0.5.
- **The v15 non-metal fixture assertion:** `TestWeightAlias_olderBundleTakesCopyPath` passes on the Mac and logs `non-metal bundle (format v15)`.
- **Item 8 of the list below is answered:** a Mac user's normal path writes the metal layout. `-backend auto` resolves to `metal` before
  `prequant.EnsureCachedGIW` picks the sidecar's target, so the sidecar is `.metal.giw`, kind 7 with fused groups. The unaliased v15 scales
  arise only from a bare `prequant` (`cpu-arm64` on a Mac), an explicit non-metal target, or a `.giw` built elsewhere.
- **Binding v15 scales directly** is therefore a rare-path nicety and is not built. The record is in the task doc's 2026-10-03 F-D02 line.

- `metal/alias.go:37-40`: aliasing is on by default "since 2026-09-24, when every gate in S6's registered rule had passed (... footprint met, the memory-hog arm, Close ordering ...)". The record says: gate 1 "NOT MET" at first (`s6-alias-2026-09-24.md:131`), later "the dense term gone condition is met" with the process total "~60 MB over" read literally (`:191-194`); the hog arm ran on the 7B only and "M26 under a hog [was] deliberately not run" (`:147-148,161,284`); M26 decode was "not resolvable at this n" (`:271-273`); and the default flip is recorded as "on the owner's decision after the gates above" (`:245`). "Every gate had passed" is not what the record says.
- **v15.** Since v15, kinds 3/4/5 store binary16 scales (`decoder/serialize.go:94,1099-1116,1254-1268`; reader `:1737-1742,1809-1821`), so a non-metal-target bundle carries f16 scales. `Int4ScalesF16` is populated only from `recordF16` at kind 7 and the fused-group case (`decoder/serialize.go:1895,1970`; `decoder/model.go:176-186`), so those scales are widened and converted again into a new buffer (`metal/alias.go:267-285`), and the banner says "carries no f16 scales (written before weights format v14, or not with -target metal)" (`:298-301`), which is wrong for a v15 non-metal bundle. The rebuild advice is still right. The record's size of the copied term is "1/8 of the nibble bytes, ~390 MB on the 7B" (`s6-alias-2026-09-24.md:141-143`). `TestWeightAlias_olderBundleTakesCopyPath` pins a v12 file (`:225-228`), not this case.
- **Fix.** Reword the header; either bind v15 scale arrays directly (their addresses are in the mapping) or fix the banner text; add a v15 non-metal fixture assertion.

#### F-D03. `HiddenLast` "bit-identical by construction" (Minor)

- `metal/backend.go:891-893` says the sequential kernels are "bit-identical to the CPU reference by construction". `metal/hiddenlast_resident_parity_test.go:12-21` records the measured cosine (about 0.9991-0.9993 on gpt2) and sets the bar at 0.998 because the task doc's 0.9999 was not reachable; `metal/prompthidden_resident_parity_test.go:17-22` records 0.99985 for CPU int4 vs Metal int4 on qwen3_5-tiny. The decode kernels are not bit-identical to the CPU's, and `forwardHiddenNoHead` returns the int8-dequantized activation (`metal/model.go:1821-1840`). Replace the sentence with the measured figure.

#### F-D04. Qwen-VL "parity verified" (Minor)

- `docs/gpu-residency-coverage.md:134-138` says Metal supports full multimodal resident decode and "Parity verified in `metal/forwardmrope_parity_test.go` and `metal/uploadkv_parity_test.go`". The former tests constant-shift invariance on `llama-tiny` or a Qwen2.5-Coder GGUF (`metal/forwardmrope_parity_test.go:33-53`), not an M-RoPE VL model; both skip when no fixture is found (`:51`). The heading "The gaps, as of 2026-09-12" (`:37`) also now contains 2026-09-17 content (`:92`). N-04's claim that `ForwardMRoPE` did not exist is closed: `metal/backend.go:539`.

#### F-G04. `cmd/gate gpu` and the vanished verify gate (Minor)

- The v0.18.0 record shows the Metal device gate reporting 9 pass / 2 skip / 11 fail on a fresh worktree until three gitignored fixture directories were copied in (`c3-metal-consumer-window-v0.18.0.md:123-133`), and one memory-state-dependent failure (`TestDenseResidentParity`: "resident context 5689 positions exceeds this backend's hard ceiling of 4096", `:138-141`). Separately it records `TestBatchedVerifyKernelParity`, "the Metal decode==verify bit-identity gate", red since `26f64807` and blocking a release (`:185-238`). At HEAD `grep` finds that test name only in the doc; `bvk*` identifiers are gone from `metal/`. The record does not say it was fixed. Whether MC3's `mc3_step_test.go` (env-gated, F-G02) took over its role is not stated anywhere I read. `cmd/gate` is not in the snapshot.
- **Status, 2026-10-01: answered.** `b51b846d` (2026-09-24) deleted the test together with the kernels it measured. The five
  `gemv_w4a8_bvk_*` kernels of the batched-verify experiment (NO-GO, `docs/completed/task-metal-batched-verify-kernel.md`) were
  never in `allKernels` or any dispatch path, so no production coverage went with them. The verify path that ships is another
  one: `--spec ngram` verifies with ForwardN on the MC3 step kernels, and its decode-equals-verify gates are
  `TestSpecVerify_forwardNMatchesForward` (logits and per-layer K/V, bit for bit) and `TestMC3Verify_sameSlotRowsBitIdentical`.
  In that sense the MC3 tests took over the role. Both skip unless `GOINFER_METAL_MC3=1` and a real checkpoint are given, and
  `cmd/gate` never sets that variable, so `gate gpu` does not run them; that is F-G02's gap.

#### F-N01. `HiddenLast` evicts the bound slot's conversation (Minor)

- `hiddenLastResident` calls `residentForgetIDs()` then runs from position 0 on whichever slot is bound (`decoder/embed.go:83-89`; `metal/backend.go:911-935`, `Reset` at `startPos==0`). Bookkeeping stays consistent (the bound slot's ids are cleared, parked slots are untouched), so there is no wrong output; but with MC1 slots an embeddings request destroys the live chat's cached prefix instead of using a free slot (`residentAcquireSlot`, `decoder/resident_reuse.go:234-266`, is not called). Cost: one full re-prefill of that conversation. The call also stays a per-token command buffer loop (`metal/backend.go:915-935`; old N-25).

#### F-N02. S6 file-change policy is one comment (Minor)

- `decoder/giwmap_darwin.go:24-25` says a `.giw` "is replaced by temp+rename, never rewritten in place, so a shared mapping cannot observe a concurrent rewrite". That depends on `cmd/prequant` and `internal/prequant` (not in the snapshot) and does not cover other tools. A truncated file under a `MAP_SHARED` mapping gives a SIGBUS on the next page touch (CPU or wired GPU read); an in-place rewrite changes live weights with no error. `MAP_PRIVATE` shares the first exposure. No size/mtime check at `Close` or at request time. Report as a policy statement that needs a test or a sentence in the user docs.

#### F-B01. P is rounded to half in steel (Minor, MLX analog)

- `metal/prefill.go:644-650` sums the unrounded `p` into `lRow` and feeds `half(p)` to the PV MMA. MLX keeps `Stile` in `AccumType` (float) and runs `MMAFrag_acc_t::mma(Otile, Stile, Vtile, Otile)` (`steel_attention.h:473-521`), so numerator and denominator use the same f32 values. The rounding error is bounded by 2^-11 relative per term, random in sign, and the pooled gate passed with it (`metal-prefill-attn-2026-09-27.md:96-107`). This is the same class as every other non-bit-identical Metal kernel. Probe: rerun set A with P kept in f32 (mixed float x half `simdgroup_multiply_accumulate`); a pooled-KL difference under the gate's noise ends the question. Not proposed as a speed item.

### (d) Carry-forward of the Sep 12 entries

Status uses the five values from the brief. "Record only" means the cited target is not in the snapshot, so I could verify the code but not the test or doc.

| ID | Status | Evidence at HEAD |
|---|---|---|
| C-01 | CLOSED-VERIFIED | `metal/model.go:1300-1314` pads the allocation to 8 rows; `metal/attention_prefill_fused_cachepad_test.go:14,31` asserts `Buffer.Len`. The steel kernel only loads keys `< jEnd` (`metal/prefill.go:602-605`), so it does not rely on the padding. |
| C-02 | CLOSED-VERIFIED | `metal/backend.go:905-907` declines `HiddenLast` on a paged MoE (also `:569`, `:842`, `:869`); chokepoint panics at `metal/moe.go:794`, `metal/gemma4_moe.go:434`; test `metal/c02_paged_forward_entrypoints_test.go:28`. |
| C-03 | CLOSED-VERIFIED (code; test record only) | `decoder/lora.go:165-196` `validateComputeTimeDims`, called at `:380`. `decoder/*_test.go` not in snapshot. |
| C-04 | CLOSED-VERIFIED | `metal/lora.go:173-176,203` (`bound` flag, deferred release); `metal/lora_bind_leak_test.go:20`. |
| C-05 | CLOSED-VERIFIED | aikit `metal.go:988-992` pins the thread in `Run1DBatchTG`. |
| C-06 | CLOSED-VERIFIED | aikit `visionmetal/encoder.go:115-127,174-181`; `visionmetal/threadgroup_budget_test.go:21-38`. |
| G-01 | CLOSED-VERIFIED | `metal/moe_model_test.go:326` sets the floor to 0. |
| G-02 | PARTIAL | Missing-cell Fatalf present (`metal/prefill_gate_ref_test.go:290-299`); the "resident not built -> Skipf" half is still there (`metal/prefill_gate_ref_test.go:182`, "metal resident not built for this model") and was never reproduced, per the record. Unchanged since the record. |
| G-03 | PARTIAL | Record says the fixture now mixes sliding and full layers; the checkpoint is gitignored and the generator absent from the snapshot, so not verifiable here. The kernel a real Gemma 3 reaches at hd=256 (`attention_prefill`, exact) has no hd=256 test at any length (F-C02). |
| G-04 | PARTIAL | Golden still re-bakes with `GOINFER_UPDATE_GOLDENS` and no argmax/cosine gate (`metal/snapshot_golden_test.go:222-225`); `metal/kernels.go:135,180` still cite `scripts/autoresearch_rmsnorm_results.tsv`, absent. |
| G-05 | OPEN (REVISIT) | `pagecost_sharedevent_test.go` comment still reads "recovers ~0% ... synchronous Metal MoE paging is not viable" (dense 1.5B, ~0.26 ms/boundary). REVISIT premise: measured on a dense shape where the boundary costs 0.26 ms, not the ~15 ms the paged 26B shows (`metal/residency_probe_test.go:11-12` per the old audit). Cheapest probe: re-run on the paged 26B (area D). |
| G-06 | CLOSED-VERIFIED | `metal/close_leak_test.go:162-199,240-300` assert `CurrentAllocatedSize`; `metal/alias_fixtures_test.go:144-168` also. |
| G-07 | CLOSED-VERIFIED | `metal/prefill_gate_test.go:75` sets the floor to 0; `TestMetalPrefillDivergenceRate` is cited only in comments that say it no longer exists (`metal/prefill_gate_test.go:48,186`, `spec_*_test.go`). `docs/measurements/prefill-gate-l1-2026-09-05.md:66` still cites it as a name (a dated record). |
| G-08 | PARTIAL | `prefill_startpos_test.go` exists and passes for hd=16 (fused). Steel at hd=128 with `startPos>0` is not covered (F-G01). |
| G-09 | OPEN | `metal/moe_prefill_measure_test.go:33` skips without a checkpoint ("a manual measurement, not a CI gate") and `:54-55` still Fatals on a paged resident. Its comment still says default-ON "above the 512-token floor" (`:17-18`); the floor is 64 (`metal/backend.go:772`). |
| G-10 | CLOSED-VERIFIED | Pointer resolved by C-06's pre-check (above). |
| N-01 | CLOSED-VERIFIED | Superseded callouts and labelled figures at `benchmarks.md:575-580,602`. New staleness layer: F-D01. |
| N-02 | CLOSED-VERIFIED | `benchmarks.md:1068-1070,1083-1087`. |
| N-03 | CLOSED-VERIFIED | `benchmarks.md:1148`. |
| N-04 | CLOSED-VERIFIED | `gpu-residency-coverage.md:25-28,102-109,134-138`; `ForwardMRoPE` now exists (`metal/backend.go:539`). New overstatement: F-D04. |
| N-05 | CLOSED-VERIFIED (record only) | `audit-2026-09-10.md` and `docs/completed/` not in snapshot. |
| N-06 | CLOSED-VERIFIED (record only) | `task-autoresearch-loop.md` not in snapshot. |
| N-07 | CLOSED-VERIFIED (record only) | `docs/completed/` not in snapshot; `grep` finds no `bvk*` code, consistent with the NO-GO. |
| N-08 | CLOSED-VERIFIED | `grep` for the old phrases finds none; `metal/backend.go:887-894` rewritten. `metal/spec_prefill_regression_test.go:43-60` rewritten. `metal/spec_verify_curve_test.go:129` still says "declined by default" in a perf-probe sentence (Minor). |
| N-09 | CLOSED-VERIFIED | no "greedy decode skips" text found. |
| N-10 | CLOSED-VERIFIED | no "11 dispatches vs 19" text found. |
| N-11 | CLOSED-VERIFIED | "Dense residency only" absent; `decoder/features.go:417` cites `metal/moe.go:474-475`. |
| N-12 | CLOSED-VERIFIED | `prefill-gate-l1-ref-b-2026-09-09.md:14`. |
| N-13 | CLOSED-VERIFIED | `metal/snapshot_golden_test.go:75-77,146`. |
| N-14 | PARTIAL | Floor still 0.95 (`metal/lora_resident_parity_test.go:114,203`) against measured 0.9998/0.9988 (`:104-111`); parked on purpose. A dropped projection could still pass. |
| N-15 | OPEN | `metal/prefill.go:1117-1125` still `make`s and zero-fills ~15 scratch buffers per call; chunked record notes "about 15 MB allocated and zeroed per pass" (`chunked-prefill-2026-09-27.md:111-113`). Now matters per chunk. |
| N-16 | OPEN | `ensurePrefill` is still called inside `PrefillLast` (`metal/prefill.go:1091`); the N-47 latch is present (`:815-819`). |
| N-17 | CLOSED-VERIFIED | `gemm_w4f16` gone; `gemv_w4a8_sa_amax` documented in `metal/kernels.go:9-16`. |
| N-18 | CLOSED-VERIFIED | subsumed: fused kernel no longer the hd=128 path (R19); fused still serves hd < 128 (area A). |
| N-19 | OPEN | `rope_f16` dispatched separately for Q and K (`metal/prefill.go:1283-1284`), no fusion. |
| N-20 | CLOSED-VERIFIED | `int4DirectBytesOnly` used in `metal/gemma4_moe.go:292-293`. |
| N-21 | CLOSED-VERIFIED | `metal/expertpool.go:213-218` uses `gpu.NewBufferLenOf`. |
| N-22 | OPEN (area D) | superseded by M-11; not re-read. |
| N-23 | CLOSED-VERIFIED (record only) | not re-read (area D). |
| N-24 | NEGATIVE-CLOSED | deliberate design as recorded; not re-read. |
| N-25 | PARTIAL | Comment rewritten (`metal/backend.go:891-899`); still one command buffer per position (`:803-823`). |
| N-26 | CLOSED-VERIFIED | `decoder/residency.go:60-69` states Metal's `ForwardBatch` is one command buffer, per-token only for paged MoE. |
| N-27 | NEGATIVE-CLOSED | Premise unchanged (`spec_optfwd.go` reliance recorded); not re-read. |
| N-28 | OPEN | Sampler/GPU overlap unchanged by anything I read; not an F item. |
| N-29 | CLOSED-VERIFIED | `bytesToU32` at `metal/model.go:558`; `int4Concat` pre-sized (`:452-677`). |
| N-30 | PARTIAL | Stale K<=1536 comment corrected at `metal/kernels.go:397-400`; `swiglu_quant` double evaluation stays a recorded negative. |
| N-31 | OPEN | `WaitDone` still captures timestamps (aikit `metal.go:740-767`); microseconds. |
| N-32 | PARTIAL | Comment fixed (`metal_vit.go:1149-1150`); the library-wide fast-math-off compile remains (`metal_vit.go:363`). |
| N-33 | CLOSED-VERIFIED | `metal/expertpool.go:66-77`. |
| N-34 | OPEN | informational; not re-read. |
| N-35 | CLOSED-VERIFIED | `decoder/model.go:1504-1535` keyed set. |
| N-36 | CLOSED-VERIFIED | `decoder/residentneed.go:62`. |
| N-37 | OPEN | micro-bench shape note; unchanged. |
| N-38 | OPEN | first-call compile note; unchanged (and `ensurePrefill` is still lazy, N-16). |
| N-39 | PARTIAL | Comment fixed; the underlying gap stays: `decoder/model.go:1783` still requires `prefillFrom == 0` for the resident path with an adapter. |
| N-40 | OPEN | `benchmarks.md:1026` still "4-bit both sides" with no int8-head qualifier. |
| N-41 | PARTIAL | The ceiling moved from 4096 to 32768 (`metal/model.go:22-35`) and an unpinned load defaults to 4096 (`:40-42`), so the 5689 case is now accepted. The planner still does not know the ceiling (`decoder/fitplan.go:233-243` has one for WebGPU only): an auto-pinned context above 32768 is refused and the model declines to the CPU path (`metal/model.go:52-57`). Tests work around it by pinning (`metal/prefill_gate_ref_test.go:167-173`, `metal/decode_decomp_test.go:80-83`). |

### (e) Checked and found correct

- **R19 steel, `metal/prefill.go:533-672`.** Ragged K: keys `>= jEnd` are staged as zero (`:513-524`) and masked by `js >= myKeys` (`:544-547`); ragged M: `min(myRow, M-1)` clamps (`:483,487,492-494`), store guard `myRow < M` (`:580`), padded Q rows exist because `Mpad = ceil8(M)` and `qkvF` is `Mpad*qkvDim` (`:911,936`). Causal boundary: `edge` is true unless the whole block lies below the simdgroup's first row's key limit (`startPos+r0+1`), a conservative superset (`:542`); the window lower bound is handled in the same predicate and in the block range (`:484,490,495,527`). Fully masked rows: `-INFINITY` with explicit `mNew > -INFINITY` and `s > -INFINITY` guards (`:555-559`), output 0 when `lRow == 0` (`:581`). GQA: `kvh = qh/(nH/nKV)` (`:476`). Barriers are threadgroup-uniform (`jStart`/`jEnd` do not depend on the lane; skips are inside the block, barriers at `:525,577`). Q/K/V positions: scatter runs before attention (`:1077` then `:1080`). Heterogeneous geometry is excluded by `prefillOK` (`metal/model.go:1027`), and sinks/qGate/DeltaNet by `prefillFeatures` (`:92-108`, no `FeatAttnSink`).
- **MLX comparison for those guards.** MLX uses function constants `align_Q`/`align_K` (`steel_attention.h:11-12,235,299,365,455,551`) so aligned shapes skip the tail code; goinfer evaluates one `edge` predicate per block at run time and zero-fills instead, which costs a branch and two compares per block, no correctness. MLX masks with `finite_min` (`:368,389,412`), goinfer uses `-INFINITY` with the explicit guards above, so the all-masked-row case is handled rather than relied on. MLX's `kb_lim`/`kb_min_causal` (`:282-292`) correspond to goinfer's `jEnd` and the `sEnd`/`sStart` simdgroup test. MLX supports sinks (`:274-279`); goinfer excludes them from prefill rather than ignoring them.
- **R17 `attention_fa_blk`, G=6,7.** Selection `metal/model.go:1511-1520`; fixed split 16 and depth floor 1536 (`:2507,2517`); window, sinks, qGate, kvI8, LoRA excluded in `canUseAttnFA` (`:2555-2571`) and in the batched twin (`metal/batch.go:483-499`); `metal/attn_fa_blk_test.go:147` is a default-runnable float64 reference, with `:49` pinning that the graded kernel is the one dispatched and `:196` pinning selection.
- **R18/R18b.** `gemvRowsFor` tiling guard (`metal/model.go:2650-2657`), per-layer minimum for qkv (`:1297-1307`), down only when `I%4==0` and K bytes fit threadgroup memory (`:1305`), pipelines named by R (`:1308-1317`).
- **f16 scales.** One conversion (`decoder.F16Bits`, `f32ToF16` calls it, `metal/alias.go:222-223`); writers store exact binary16 (`decoder/serialize.go:1109-1116`); every kernel I read takes `device const half*` scales (`metal/kernels.go:344,358,364`); upload paths (`int4Buf`, `int4Concat`, `int4ConcatA`, expert pool precompute) all go through that function.
- **S6 alias.** Page-aligned windows over `MmapAliasWindow` (`decoder/model.go:192-201`), 16-byte offset check and decline (`metal/alias.go:209-212,246`), PROT_READ `MAP_SHARED` mapping (`decoder/giwmap_darwin.go:43`), fork exclusion (`decoder/model.go:188-190`), Close ordering and device-size flatness tested (`metal/alias_fixtures_test.go:144-168`, `s6-alias-2026-09-24.md:221-224`), logits byte-identity over 23 fixtures with a mutation check (`:207-213`).
- **MC1 slots.** `kvContig` single allocation with per-slot padded region (`metal/model.go:1315-1353`); bookkeeping swap in `residentBind` (`decoder/resident_reuse.go:271-290`); recurrent-state families keep one slot (`:212-218`); `pickResidentSlot` avoids truncating another conversation for a shared preamble (`:292-323`).
- **MC3.** `batchIneligible` (`metal/batch.go:300-333`) excludes kvI8, sandwich, layer norm, qk-norm, windows, MoE and untiled widths; adapters are refused (`metal/batch.go:524`) and held out by the exclusive claim (`decoder/mc3_batch.go:156-163`).
- **PrefillLast hygiene.** Panics from compile and the per-call allocations are recovered into an error (`metal/backend.go:834-843`); `startPos < 0` and cap are checked (`:741`).
- **Gemma.** `TestGemma1ResidentParityMetal` uses `Fatalf` when the fixture does not go resident rather than `Skip` and gates at cosine 0.999 on 16 positions (`metal/gemma1_resident_parity_test.go:18-55`); `TestGemma2DeclinesResidentMetal` asserts the decline reason names the attention softcap (`:59-72`), which is the kind of assertion CLAUDE.md asks for. `capability-matrix.md` has no Metal column; its `GPU-resident` column (lines 113-114) is consistent with these two tests.
- **Decode attention depth.** `metal/attn_shape_test.go:37-91` compares the shipped decode kernel with a CPU reference at hd 128/256/512, windows, and nKeys 4096, 8192, 16384, 32768, by default.
- **Executor race.** `attnPlan` is compared per job and re-encoded on mismatch (`metal/model.go:2837-2859`).

### (f) Questions I could not settle statically

1. Whether `--kv i8` really reaches `PrefillLast` in a served Metal run. I found no guard in `backend.go`, `model.go` or `decoder/model.go:1554-1596`; I did not read every wrapper between `serveapp` and the resident. The probe in F-C01 settles it in one run.
2. Steel numerics at `startPos > 0`, with windows, and at ragged M/K for hd=128. No evidence either way.
3. Whether chunked prefill is still bit-identical to whole prefill on the steel build. Argued from structure (F-G02), not measured.
4. How often the fit guard auto-pins between 4097 and 32768 (F-C02) or above 32768 (N-41) on a 16 GB Mac. It depends on free RAM at load.
5. Whether a plain `go test ./metal/` is green. The 98 default files depend on `testdata/` fixtures that are gitignored; some tests `Fatalf` when a fixture is missing (for example `metal/gemma1_resident_parity_test.go:21-23`) and others `Skip` (`metal/forwardmrope_parity_test.go:51`), so the result on a clean checkout is red for one reason and green-by-skip for another, as the v0.18.0 record shows (`c3-metal-consumer-window-v0.18.0.md:123-133`). Also unclear whether `testdata/gemma1-tiny` and `gemma2-tiny` are generated or committed: the name appears only in the test files in this snapshot.
6. `cmd/gate gpu` wiring (no `cmd/` in the snapshot): which tests it names, and whether it sets `GOINFER_METAL_MC3=1` or `GOINFER_HEAVY_TESTS=1`.
7. What happened to `TestBatchedVerifyKernelParity` (F-G04).
8. Whether `prequant`'s default `-target` writes the kind-7 form, which decides whether the v15 aliasing gap (F-D02) is common (`internal/prequant` not in the snapshot).
9. Whether `.giw` replacement is always temp+rename (F-N02).
10. Whether any registry architecture has hd=128 with nH/nKV > 8 and a dense layer (F-C03).
11. Whether the R17/R18 end-to-end numbers survive a pre-registered re-run; the headline sentences rest on same-session A/Bs by the row's own description (`benchmarks.md:47`).
12. NAX / Metal 4: nothing in this area depends on them; `*_nax*` files in MLX were not read.

---

# Part III — Program, closed items and carry-forward

## 10. Program

Ordered by effect on a shipped shape per unit of work. Every performance item follows the repo's rules (`red-october.md`
§6): a pre-registered band, a discriminating probe, a kill line, and measure-don't-assume — a projection here is a prior,
not a result. Budget tags: **[day]** is a code edit, a unit test or a short in-sequence A/B; **[night]** is a
pre-registered sweep, a pooled fidelity gate or a multi-hour real-model run. The tags are a triage, not an estimate. Never
`git add -A`; one commit per item.

### Track 0 — guards and text (no numeric change; one commit each)

| # | Item | What | Status, 2026-10-01 |
|---|---|---|---|
| T0.1 | A-C01 ≡ F-C01 | `&& !r.kvI8` in `prefillOK` (`metal/model.go:1027-1029`), so `PrefillPath()` reports sequential; a test that builds an i8 resident and asserts `PrefillLast(embs[≥64], 0)` returns the decline error with the floor knob at 0. Then reproduce the failure once on the Mac (llama-tiny, `KVPrecision:"i8"`, `PrefillLast` of 16 embeddings against 16 `Forward` calls) to confirm the derivation. [day] | **Done** (`de1c7f17`): the guard and `TestPrefill_declinesInt8KV`. The device reproduction of the wrong K/V is not in the record |
| T0.2 | F-C02, F-G03 | Decline to the sequential path when `!useFusedAttn && startPos+M > attnScoreTileBound` (beside `metal/backend.go:817`); replace `TestMetalCtxCapWithinKernelBound` with a prefill run at nKeys = 4097 on an hd=256 fixture (and an hd=64 fixture with `GOINFER_METAL_FUSED_ATTENTION=0`); correct the "without threadgroup memory overflow" claim at `metal/model.go:28-29`. [day] | **Done:** the decline and `TestPrefill_exactAttentionDeclinesPast4096Keys` (`de1c7f17`); F-G03's replacement, `TestAttentionKernelsPastTileBound`, and the `metal/model.go` comment (2026-10-01) |
| T0.3 | F-C03, C-N01, D-C01, A-C02 | `canUseAttnFA`/`canUseAttnFAAt` return false when `nH/nKV > 8` (arrays are `ATTN_FA_MAXG 8`); a length check in aikit's `Encoder.Dispatch` binding scratch (`[16]`, widest call binds 15); a gpt-oss decline test inside the expert-major branch so a future feature-map edit fails closed; a host finite-check on `PrefillLast` logits (about 0.05 ms) that returns a decline error so the sequential path re-runs the prompt. [day] | **Done** except C-N01 (2026-10-01): `attnFAGroupOK` keeps groups over `ATTN_FA_MAXG` off attention_fa; an explicit gpt-oss term in `prefillOK`, tested with every gpt-oss feature admitted to the prefill map; a finite check that declines `PrefillLast`. C-N01 is in aikit and waits for its next release |
| T0.4 | C-C01, N-41 | Give `Plan("metal")` and both fit guards a backend ceiling (the WebGPU shape, `decoder/fitplan.go:233-243,275`), price Metal KV as f16 through `ResidentKVBytes("metal", …)`, and make an auto-pin a ceiling Metal clamps to rather than refuses. First a unit test with injected `hostRAMAvailable` and a 131072-window config. [day] | **Done** (2026-10-01), the guards excepted: `resolveMetalCtxCap` treats an auto-pin as a ceiling that may lower the 4096 default but never raise it, and `Plan("metal")` has Metal's ceiling (4096 unpinned, 32768 with `-ctx`; `decoder.MetalCtxDefault` and `MetalCtxCeiling`, which `metal/model.go` now takes its constants from). The guards still price the CPU's KV, because they run before the backend decides whether it can host the model, and a model Metal declines runs on the CPU, whose KV grows to the request. The cost: a machine too tight for the 2048-position floor at f32 is refused a load Metal could run at f16. The probe's kill line did not fire: with memory for 40000 f32 positions the guard pinned 40000, which Metal refused before the fix |
| T0.5 | Docs | One pass over `docs/benchmarks.md` for A-D01, B-D01, F-D01 and D-D01 (`:43`, `:47`, `:1026-1031`, `:1080-1081`, `:1976-1985`, `:2017`, `:2260-2262`: dated callouts, no number invented; "AHEAD in every cell" becomes "every 1.5B and 7B cell" until the 0.5B cells are re-run); `red-october.md` §2.6, R11(c) and the `:91` row (D-D01 1–3); the `alias.go` header and v15 banner text (F-D02); the `HiddenLast` doc sentence (F-D03); `gpu-residency-coverage.md:134-138` (F-D04); the stale comments in A-D02, B-D02, C-D01, D-D01 4–10, E-D01; restore or regenerate `r3-startpos-speed-2026-09-21.md` (A-D03). [day] | **Done** (2026-10-01), except F-D02's code options (binding v15 scales directly, a v15 non-metal fixture assertion). The headline Metal rows were rewritten from the 2026-09-30 peer sweep earlier (`aff6f5e8`, `8e73aef5`); every other line and page listed is now re-checked, with dated callouts and no number invented. Two findings did not hold: A-D03's record exists (`fdd990ba`, the snapshot lacked it), and C-D01 item 3 read this Mac's M1 Pro as 16 GPU cores, where `system_profiler` gives 14, the value `attnFACoreCount` has |

### Track 1 — probes that decide Majors (little or no code)

| # | Probe | Decides |
|---|---|---|
| T1.1 | Rerun `TestMC5_passCost` on the current build at startPos 64, 2048 and 8000, C = 16/32/48/64; read (startPos 2048) − (startPos 64) at C=32. ≤10 ms: A-P03 closes; ≥25 ms: build the BQ=16 steel variant. Zero code. [night] | A-P03 |
| T1.2 | Test-only `attnFAFloorOverride` read inside `attnPlanFor` and `canUseAttnFA` (next to `attnFASplitOverride`), then interleaved legacy-versus-blk arms at nKeys 256, 384, 512, 768, 1024, 1280, 1535, 5 reps × 20 tokens, on the 1.5B and 7B. [night] | B-P03 |
| T1.3 | Log `kernStart(t+1) − kernEnd(t)` from the existing timestamps (`ReadTimes`) over 200 tokens on the 1.5B and 0.5B. Median under 0.15 ms kills C-B01. [day] | C-B01, C-P02 |
| T1.4 | `gemv_w4a8_sa_rows4` at fixed weight bytes (~15 MB) at K=1536, 2048, 3072. A time-per-byte advantage of 10% or more for K=2048 confirms the idle tail. [day] | B-P04, B-P06 |
| T1.5 | Rerun `TestMC5_prefillChunkInvariance` (and `passCost`) on the steel-wired build; then add a default-run version on a generated hd=128 random-weight fixture with two slots. [night, then day] | F-G02 |
| T1.6 | Resident set after load on the 7B with `--kv-sessions 1` and `4`; whether untouched KV pages count. [day] | E-P09, C-C01 |
| T1.7 | In-sequence attention at nKeys 1536..6144 in 128-key steps, both models; kill if max/min per-key cost is under 1.15×. [night] | B-P08 |
| T1.8 | `vmmap` region table on the aliased v15 M26 at token 32 to attribute the 1,948 MB heap. [day] | C-P01 |
| T1.9 | `GOINFER_MOE_PROF_SPLIT=1` on the aliased v14 M26, and an N=64 rerun under the kill-watch (the fork-collapse fix of 2026-09-24 postdates the failures). [night] | D-P02, C-B03, M-11 |
| T1.10 | Drive `forwardMultiInto` over a 32-token prompt in 8-row pieces; compare all K/V bytes and the last row's logits against the sequential loop (expect zero differences); time K = 8/16/32/64. [day] | E-P01 |
| T1.11 | One served K=3900 TTFT cell against Ollama (R19 owes it); one 0.5B depth cell at 2048 and 3900 (B-D01). [night] **The served K=3900 cell ran 2026-09-30: LEVEL, 0.983 (peer sweep cell h). The 0.5B depth cells are owed.** | A-D01, B-D01 |
| T1.12 | Count `CePad` per active expert on a real MoE; microbenchmark `gemm_w4f16_store` at 16, 32, 64 rows on an expert shape. [day] | D-B02 |
| T1.13 | Requantisation error of the real gpt-oss checkpoint (CPU script): per-group max-element distribution, relative RMS, collision rate. [day] | D-B03 |
| T1.14 | A same-session three-way decode run (goinfer, mlx-lm, Ollama) on the 1.5B and 7B at depth 128, 2048 and 3900, to replace the cross-session 0.82× and 0.75× with a ratio that can be quoted; the box's recorded drift is about 3.5%. [night] | §0, B (f)9 |

### Track 2 — builds, in order

1. **E-P01, short prompts through the 8-row step.** Bit-identical, so no fidelity gate; the route exists
   (`PrefillLastNArgmax` runs consecutive same-slot positions through `forwardMultiInto` in pieces of 8, identity
   gate 0 differing logits and 0 differing K/V elements in 7 cases on both models). Needs a mode that skips the LM head on
   non-final pieces (about 8% per piece). Gate the route to K ≤ ~40 until measured. Band: K=32 305→70–85 ms, K=16
   167→~40 ms, 7B K=32 ~1.06 s→~0.28 s. Kill: K=32 TTFT above 0.6× sequential, or slower than the batched pass at K ≥ 16
   once A-P01 lands. Also gives `--exact-prefill` users about 3.7× at every K. [night]
2. **Decode attention coverage, one campaign on one harness** (`TestMetalDecodeDecomp` swap, interleaved, 5 reps × 20
   tokens, in-sequence attention = full minus no-op): B-P03 (floor; probe T1.2), B-P02 (g4/g5/g8, per-head equality
   against g7 first), B-P01 (hd=64 blk, S swept in {8, 16, 24, 32} because R17 showed the response is not monotone), then
   B-P08's sweep to see whether the fixed S=16 sawtooth needs a rule. R17's registered kill line applies (in-sequence
   ratio under 1.5× at both depths). Every kernel change here goes through the set-B fidelity gate, and
   `TestAttnFABlkIsTheGradedKernel` (SHA-256 of the blk region) means any edit to blk first has to be shown bit-identical
   to the graded kernel. [night]
3. **Small-M prefill: A-P01 then A-P02.** BM=32 for `Mpad ≤ 32` and BN=32 for `N ≤ 2048 && Mpad ≤ 64`, both bit-identical
   (each output accumulates K in the same ordered 8-wide chunks); extend `TestMC5_prefillChunkInvariance` to C in
   {16, 32, 48} so it crosses the selector. Ship at ≥1.15× pass wall at C=32 and zero differing bits. Then the pooled-gate
   cells at K=16/32/48 for A-P02 (needs CPU reference files for those K). Re-evaluate the decoder's `suffix ≥ 8` switch
   (A-P05) afterwards; no change before. [night]
4. **Decode GEMV residue: B-P04, then B-P06, then B-P05.** B-P04's lane-balanced form is bit-identical (exact integer
   exchange plus an owner-lane float chain): prototype in `gemv_r18_test.go`, byte-compare through `TestR18InSequence`
   (zero differ, as in R18b), grade in sequence on the 1.5B. Bar: the owner's standing line, park under 1.02×. B-P06's
   sweep (threadgroup {64,128,256,512} × R {2,4} × {staged, device-read}) shares the harness. B-P05 only with the MC3
   down kernel moved in the same change and the fidelity gate run; the 7B cannot stage K×2 B as half (37,888 B exceeds
   the 32 KiB budget), so it needs int8 staging or a two-chunk K loop. [night]
5. **C-B01, on-device token chain, for greedy and device-sampled modes.** Reuse `gemv_w8a8_amax` + `argmax_finish`, add a
   gather kernel that writes the next embedding row (with the arch embed scale and any learned position add, exactly as
   `loadEmbedRow` does), commit CB(t+1) behind CB(t) on the same queue. Bit-identical. One speculative forward past an
   EOS or stop string is wasted per sequence. C-P02 (a sample mode on `execJob`) is the smaller sibling. [night]
6. **MoE and hybrid prefill, in this order:** D-B01 (T-loop recurrence kernel plus batched conv, gates and L2-norm, on the
   `metal/deltanet_test.go:71-72` geometry; D-B05's lane-per-Dk layout is a prerequisite for register state, since 128 f32 per
   thread in today's thread-per-row layout is too many), D-B02 (device offsets and a tile table, bm=16), D-P01 (paged
   expert-major through the pool; kill if under 2× at M=512 with N=8, because then pread, not dispatch count, is the
   cost), D-P04 and D-P03 for resident MoE decode, D-B04 (R18's rows form on the shared expert and DeltaNet projections,
   bit-identical). The first two need D-G01's gate to exist first. [night]
7. **The batched step:** E-P03 (instantiate fb4 for `mc3_bt`/`mc3_btd`, per model at `buildBatch` like `calibrateRows`; FB
   only changes which rows a simdgroup owns, so bit-identical), E-P02 (adjacent-threadgroup layout for 2-row 7B
   gate|up with the production body), E-P05 (multi-row `attention_fa_blk` and combine kernels by the existing `edit`
   method), E-P06 (calibrate the verify cost curve at load), E-P08 (greedy rows through the existing draw path), E-P07
   (QK-norm rows kernel, so Qwen3 gets MC3). [night each]
8. **Memory:** C-P01 (pread the scale span with the nibbles into the slot, +12.5% pread bytes, and delete both scale
   caches; bit-identical), E-P09 (lower the default slot count if untouched pages count against resident memory), F-D02
   (bind v15 scale arrays directly for non-metal-target bundles).

### Track 3 — gates

F-G01 (default-run float64-reference test for steel: hd=128; M in {1, 7, 31, 33, 100}; startPos in {0, 5, 16, 1000};
window in {0, 64}; nH/nKV in {12/2, 14/2, 32/8}; K/V allocated to exactly startPos+M rows rounded to 8, not page-rounded;
mutation-check by shifting the causal limit by one). F-G02 (above). D-G01 (per-layer cosine of the MoE FFN output against
the sequential path on a real MoE at M=64/512, a router-flip count, and one mutation each for scatter weights, route tie
order and shared expert on/off). E-G01 (read the CI configuration; run `go test ./metal` on a Mac without the env and
count skips; promote a small identity subset to default). A-G01 (a floor-predicate test: `startPos+M < 64` declines,
`startPos=1000, M=8` admits, the knob parses; an hd=128 case against the exact kernel). D-G02 (the two DeltaNet kernels
into the chain gate). C-G01 (the RSS-only free test also asserts `CurrentAllocatedSize`). F-G04 (find where
`TestBatchedVerifyKernelParity` went and say whether MC3's step test replaced it).

### Track 4 — owner decisions (each changes a contract, so none is proposed as a default)

- **O1 (A-P04).** Embeddings: exact or batched. Bit-identical pipelining first (1.05–1.5×); a batched trunk (~10× at
  K=512) gives up exact embeddings, which CUDA and the CPU path keep on purpose, and needs a cosine gate against today's
  `HiddenLast`.
- **O2 (D-B05).** `delta_rule`'s summation order. A lane-per-Dk layout is not bit-identical; the stage gate is cosine ≥
  0.999999, so the bit-identity language is a self-imposed rule that mirrors CUDA.
- **O3 (B-N01, §3 note 1).** Whether Metal takes an int4 head. The existing SA rows kernel at N = V would serve it; lossy
  (about 2.3 points top-1, quality re-eval parked); worth 4.4% (7B) to 6% (1.5B) of the token. Metal stays default-off
  either way.
- **O4 (B-P05).** Moving decode's down projection off the per-word chain changes bits and, unless MC3 moves too, breaks
  decode/verify agreement.
- **O5 (D-B03).** Native MXFP4 against requantised int4; the 0.99843 oracle gate moves.
- **O6 (E-P04).** A device top-k/top-p/min-p path (about 13% of a 1.5B token on filtered requests, counted); the CUDA
  K-best design keeps the host draw bit-identical when K covers the mass.
- **O7 (F-B01).** Keeping P in f32 in steel (MLX does); fidelity only, not a speed item.

### What the bands add up to

A sum of the upper ends of bands that overlap, for orientation only, never a forecast:

| Shape | Items | Band |
|---|---|---|
| 1.5B decode, depth 128 | B-P04, B-P05, C-B01, B-P09 (B-P06 overlaps B-P04 and is left out) | 5–21% of token time |
| 7B decode, depth 128 | B-P04, B-P05, C-B01 | 2–14% |
| 1.5B decode, 1024–1535 keys | B-P03 | 5–20% |
| 0.5B decode, 2048–3900 keys | B-P01 | 1.3–1.85× |
| Decode on ungraded group sizes, 2048+ keys | B-P02 | 1.15–1.4× |
| 1.5B/7B decode at 3900 keys | B-P07, B-P08 | 0–4.6% and a mean ~2.5% staircase |
| Fresh prompt of 32 tokens | E-P01 or A-P02 | 305→70–120 ms |
| Paged M26, 512-token prompt | D-P01 | 3–15× TTFT |
| Resident hybrid prompt | D-B01 | 3–8× TTFT |
| M26 footprint | C-P01 | −1.43 GB |

Not worth a probe before the items above: B-P09 (0.3–2%), E-N01 (≤0.9%), C-N03 (≤0.5%), A-B02 (1–3% of a pass), E-P10
(~1%, opt-in), A-B01 (not recommended), B-P07 (0–4.6%, needs a tree-order gate and a hash update), B-P08 until T1.7
says the staircase is real.

---

## 11. Not proposed, because the record or this audit closed them

The Sep 12 list and its status today:

| Sep 12 item | Status |
|---|---|
| Split-KV / dedup attention variants | **Superseded for decode:** R17's `attention_fa_blk` is a split-KV decode kernel (S=16), shipped. The closure concerned decode attention at one query row; A-P03 is a different kernel at 9–64 rows |
| Stage-B GEMV | Holds. R18/R18b are a different technique (rows per simdgroup, masked half-staged), graded in sequence |
| ICB, unretained references, megakernel, dispatch-count fusion | **Hold**; C recomputed each premise on today's kernels (host side after encode-ahead is 0.45 ms; GPU floor 0.86 ms is unreachable from the host; qkv and gate\|up are already fused; 10% fewer dispatches is at most 0.67% of a 1.5B token) |
| rope2+kv_store | Holds (0.6%, unwired `rope2_kv`) |
| sa_qv | Holds |
| Batched small-M verify | **Superseded:** shipped as MC3's step-kernel verify (2026-09-27) |
| Fused argmax | Holds for single-token decode; **REVISIT for MC3 rows only** (E-P08, ≤2.9% at B=8) |
| `WILLNEED` | Holds |
| Sampler overlap above temperature 0.2 | **Superseded:** R7b's device sampler for temperature-only; filtered sampling is E-P04, and the executor half is C-P02 |

New in this audit, MLX techniques examined and not recommended:

- **MLX's `qmv_fast` form** as a replacement for R18b: ALU parity, goinfer issues fewer loads (B (e)). Only the lane
  mapping idea survives, as B-P04, in a bit-identical form.
- **`qmm_splitk`** (A-B01): reorders each element's reduction; keyed on M it makes the result depend on chunk size.
- **Swizzle:** off on MLX's non-NAX path (`matmul.cpp:456`), nothing to borrow on M1 Pro.
- **MLX's per-head K reads in `sdpa_vector`:** goinfer loads K once per key for the whole group, which is cheaper.
- **`qmv_wide`** (register-shared M-row GEMV): MLX itself gates it to generation ≥15 for affine; consistent with the MC2
  S0 negative.
- **Argument buffers, `setBytes`, fewer encoders:** each leaves the transition count unchanged (C); argument buffers also
  drop implicit residency.
- **Concurrent dispatch on the dense decode chain:** no unbarriered pair exists (C, 0 to +0.3%). M-16 stays closed for
  the serial encoder.
- **M-09's staged K read:** the 0.43× negative stands for the legacy kernel; its premise no longer describes the block
  kernel (loose end: the maxAbs 1.7e38 mismatch was never root-caused).
- **NAX / Metal 4 tensor ops:** M5 only.

## 12. Carry-forward of the Sep 12 findings

One row for every ID in `docs/audit-metal-2026-09-12.md`: 16 M-, 10 C-, 10 G- and 41 N-. Statuses are CLOSED-VERIFIED (the code or test that shows it is cited), OPEN, PARTIAL, REGRESSED or NEGATIVE-CLOSED; "record only" means the cited target was not in the audited snapshot, so the code was verified but not the test or doc. A letter in parentheses names the reviewing area that supplied the row; a row without one is from area F's review. Where two reviewers gave different statuses the row says which one is used and why. No finding was found REGRESSED.

| ID | Status | Evidence and where it went |
|---|---|---|
| M-01 | CLOSED-VERIFIED | `ForwardNoLogits` `metal/backend.go:593-609`, pipelined with a `noHead` bit on `execJob` (`metal/model.go:1955-1970`), used at `decoder/model.go:1616`; byte-identical tests `metal/kvonly_prefill_test.go:19-75` (A, C). Residual: `HiddenLast` stays synchronous (A-P04). |
| M-02 | CLOSED-VERIFIED, superseded | Floor 512→256→64 (`metal/backend.go:611-626`); the remaining question is A-P02 (a gate floor, not a speed floor). |
| M-03 | CLOSED-VERIFIED, superseded by R16 | `metal/prefill.go:78-140`; R16 about 3.2× recorded (`metal-prefill-gemm-s2-2026-09-25.md`). Residue: A-P01, A-B02. |
| M-04 | PARTIAL | O in registers and the wider key tile are superseded by R19 (`metal/prefill.go:533-672`). Packing a GQA group's heads into one threadgroup is not done; MLX does not do it either (`steel_attention.h:97`) and R19's record calls it "an option neither peer takes" (A). Residue: A-P03, F-G01. |
| M-05 | PARTIAL | Non-paged generic MoE has expert-major batched prefill, default ON (`metal/prefill.go:1313-1469`); paged MoE, Gemma-4 MoE and DeltaNet still run as M decode tokens (`metal/model.go:1027-1029`): D-P01, D-B01, D-B02, D-G01. |
| M-06 | PARTIAL | `FeatPerLayerRoPE` declared (`metal/model.go:114`); Gemma 3 reaches the batched GEMM; hd=256 attention stays on the exact scalar kernel: A-B03, and its `sc[4096]` limit F-C02. |
| M-07 | PARTIAL | Row4-skip half CLOSED-VERIFIED (`decoder/weights.go:720-724,769-775,805-808,1208-1213,1522-1526`); the host-release half was declined as permanent. S6's NoCopy alias, default ON, removes the dense heap term (1.5B 90 against 1,013 MB, 7B 105 against 4,134 MB at token 32, rec `s6-alias-2026-09-24.md:188-189`). Residue: C-P01 (paged scale cache), F-D02 (v15 non-metal bundles). |
| M-08 | CLOSED-VERIFIED (code); unmeasured on a Mac | `metal/kernels.go:1971-2010` launches ceil(Out/256) threadgroups; `lora_delta_multitg_test.go` (E). No Mac adapter-decode record exists. |
| M-09 | NEGATIVE-CLOSED | Probe `attn_kread_staged_probe_test.go`: 0.43× at 2048 keys (B). Loose ends: the maxAbs 1.7e38 mismatch was never root-caused, and the premise no longer describes the block kernel that serves hd=128, G=6,7, ≥1536 keys; it still describes the legacy kernel (<1536 keys, hd=64, kvI8, windows, sinks). |
| M-10 | NEGATIVE-CLOSED, premise stale | Measured 2–6% slower on pre-R18 SA arithmetic at R=1 (B). The SA family is now 1.38×/1.46× faster at depth 128 and coal down is staged at R=4: REVISIT filed as B-P05. |
| M-11 | PARTIAL + REVISIT | Contiguous pool and `slotIdx` shipped (`metal/moe.go:881-921`, `metal/expertpool.go:213-216`). The ~14 ms boundary premise is stale (61 × 14 ms = 854 ms against a recorded 128–167 ms token); the shared-event half was never re-run: D-P02, C-B03. |
| M-12 | CLOSED-VERIFIED, cross-expert half only | Cross-expert goroutines `metal/expertpool.go:370-394`; per-expert spans `metal/moe.go:673-687`, `metal/gemma4_moe.go:356-360` (D). The per-expert 3-pread half is not done, as recorded. |
| M-13 | CLOSED-VERIFIED | `autoMoESlots` `metal/backend.go:214-244`, floor `moeTopK`, ceiling 64. |
| M-14 | CLOSED-VERIFIED (wiring); saving not re-measured | `e2.UseResidencySet` `metal/moe.go:1016`, `metal/gemma4_moe.go:594`; aikit `TestEncoder_useResidencySet`; rationale `metal/model.go:1584-1647` (C, D). REVISIT of the dense case: C-B02. |
| M-15 | OPEN; not re-audited | Outside the six areas. Re-checked only that goinfer imports neither `visioncuda` nor `visionmetal` (`docs/tasks/task-aikit-boundary-2026-09.md:33`, verified 2026-09-24) and that `docs/multimodal.md:186` still says Metal is not started. The Sep 12 reasoning for not wiring aikit's Metal tower is unchanged; `metal_vit.go` in v1.51.0 was not re-read. |
| M-16 | NEGATIVE-CLOSED (serial-encoder premise) | Untracked 99.6–101.5% of tracked (aikit `hazard_tracking_probe_test.go`). Holds for the dense chain; REVISIT for the batched step only (E-P05). |
| C-01 | CLOSED-VERIFIED | `metal/model.go:1300-1314` pads the allocation to 8 rows; `metal/attention_prefill_fused_cachepad_test.go:14,31` asserts `Buffer.Len`. The steel kernel only loads keys `< jEnd` (`metal/prefill.go:602-605`), so it does not rely on the padding. |
| C-02 | CLOSED-VERIFIED | `metal/backend.go:905-907` declines `HiddenLast` on a paged MoE (also `:569`, `:842`, `:869`); chokepoint panics at `metal/moe.go:794`, `metal/gemma4_moe.go:434`; test `metal/c02_paged_forward_entrypoints_test.go:28`. |
| C-03 | CLOSED-VERIFIED (code; test record only) | `decoder/lora.go:165-196` `validateComputeTimeDims`, called at `:380`. `decoder/*_test.go` not in snapshot. |
| C-04 | CLOSED-VERIFIED | `metal/lora.go:173-176,203` (`bound` flag, deferred release); `metal/lora_bind_leak_test.go:20`. |
| C-05 | CLOSED-VERIFIED | aikit `metal.go:988-992` pins the thread in `Run1DBatchTG`. |
| C-06 | CLOSED-VERIFIED | aikit `visionmetal/encoder.go:115-127,174-181`; `visionmetal/threadgroup_budget_test.go:21-38`. |
| C-07 | CLOSED-VERIFIED | `SetAdapter` tears the executor down because the pre-encoded t+1 buffer bakes in LoRA state; `execLoop` `metal/model.go:2091-2156` (C). |
| C-08 | CLOSED-VERIFIED | `prefillOK` excludes a paged generic MoE (`metal/model.go:1027-1029`, with its C-08 comment); paged entry points decline (`metal/backend.go:590-593,885`) (D, F). |
| C-09 | CLOSED-VERIFIED (pointer, via G-10) | Host pre-checks are the gate; `recordExecErr` sites `metal/gumbel_sample.go:69-71`, `metal/model.go:2253` (C). |
| C-10 | CLOSED-VERIFIED | The SA rows kernels' no-guard tail hazard is covered by `bad8` width checks, `model.go:~1238-1262` (B). |
| G-01 | CLOSED-VERIFIED | `metal/moe_model_test.go:326` sets the floor to 0. |
| G-02 | PARTIAL | Missing-cell Fatalf present (`metal/prefill_gate_ref_test.go:290-299`); the "resident not built -> Skipf" half is still there (`metal/prefill_gate_ref_test.go:182`, "metal resident not built for this model") and was never reproduced, per the record. Unchanged since the record. |
| G-03 | PARTIAL | Record says the fixture now mixes sliding and full layers; the checkpoint is gitignored and the generator absent from the snapshot, so not verifiable here. The kernel a real Gemma 3 reaches at hd=256 (`attention_prefill`, exact) has no hd=256 test at any length (F-C02). |
| G-04 | PARTIAL | Golden still re-bakes with `GOINFER_UPDATE_GOLDENS` and no argmax/cosine gate (`metal/snapshot_golden_test.go:222-225`); `metal/kernels.go:135,180` still cite `scripts/autoresearch_rmsnorm_results.tsv`, absent. |
| G-05 | OPEN (REVISIT) | `pagecost_sharedevent_test.go` still reads "recovers ~0% … not viable", measured on a dense 1.5B where the boundary costs 0.26 ms, not the paged M26's. Cheapest probe: re-run on the paged 26B. C-B03 (MLX's spin fence is a different mechanism), D-P02 (premise stale). |
| G-06 | CLOSED-VERIFIED for two gates; residue C-G01 | `metal/close_leak_test.go:162-199,240-300` and `metal/alias_fixtures_test.go:144-168` assert `CurrentAllocatedSize`; `TestMetal_CloseFreesMemory` still asserts RSS only. |
| G-07 | CLOSED-VERIFIED | `metal/prefill_gate_test.go:75` sets the floor to 0; `TestMetalPrefillDivergenceRate` is cited only in comments that say it no longer exists (`metal/prefill_gate_test.go:48,186`, `spec_*_test.go`). `docs/measurements/prefill-gate-l1-2026-09-05.md:66` still cites it as a name (a dated record). |
| G-08 | PARTIAL | `prefill_startpos_test.go` exists and passes for hd=16 (the fused kernel). Steel at hd=128 with `startPos>0` is not covered by any default test (F-G01, A-G01). |
| G-09 | OPEN | `metal/moe_prefill_measure_test.go:33` skips without a checkpoint ("a manual measurement, not a CI gate") and `:54-55` still Fatals on a paged resident. Its comment still says default-ON "above the 512-token floor" (`:17-18`); the floor is 64 (`metal/backend.go:772`). |
| G-10 | CLOSED-VERIFIED | Pointer resolved by C-06's pre-check (F); host pre-checks are the gate and `recordExecErr` sites exist at `metal/gumbel_sample.go:69-71`, `metal/model.go:2253` (C). |
| N-01 | CLOSED-VERIFIED | Superseded callouts and labelled figures at `benchmarks.md:575-580,602`. New staleness layer: F-D01. |
| N-02 | CLOSED-VERIFIED | `benchmarks.md:1068-1070,1083-1087`. |
| N-03 | CLOSED-VERIFIED (labelling), superseded | Superseded 2026-09-18 by `bench_peer.py` over the real server (B); decode rows come from the served path (`benchmarks.md:47`). The underlying fact, no `ResidentGreedy` on Metal (`ForwardArgmax` is test-only, `metal/model.go:2205-2252`), still holds: C-B01. |
| N-04 | CLOSED-VERIFIED | `gpu-residency-coverage.md:25-28,102-109,134-138`; `ForwardMRoPE` now exists (`metal/backend.go:539`). New overstatement: F-D04. |
| N-05 | CLOSED-VERIFIED (record only) | `audit-2026-09-10.md` and `docs/completed/` not in snapshot. |
| N-06 | CLOSED-VERIFIED (record only) | `task-autoresearch-loop.md` not in snapshot. |
| N-07 | CLOSED-VERIFIED (record only) | `docs/completed/` not in snapshot; `grep` finds no `bvk*` code, consistent with the NO-GO. |
| N-08 | CLOSED-VERIFIED | `grep` for the old phrases finds none; `metal/backend.go:887-894` rewritten. `metal/spec_prefill_regression_test.go:43-60` rewritten. `metal/spec_verify_curve_test.go:129` still says "declined by default" in a perf-probe sentence (Minor). |
| N-09 | CLOSED-VERIFIED | `finalizeLogits` runs after every logits readback and applies the softcap host-side: `metal/model.go:1856,1855-1862` (B). |
| N-10 | PARTIAL | Fixed 2026-09-13; the replacement count at `metal/model.go:2930-2934` omits the fa combine dispatch (B-D02). The 310 figure is right below the floor. |
| N-11 | CLOSED-VERIFIED | "Dense residency only" absent; `decoder/features.go:375` cites `metal/moe.go:474-475`. |
| N-12 | CLOSED-VERIFIED | `prefill-gate-l1-ref-b-2026-09-09.md:14`. |
| N-13 | CLOSED-VERIFIED | `metal/snapshot_golden_test.go:75-77,146`. |
| N-14 | PARTIAL | Floor still 0.95 (`metal/lora_resident_parity_test.go:114,203`) against measured 0.9998/0.9988 (`:104-111`); parked on purpose. A dropped projection could still pass. |
| N-15 | OPEN | `metal/prefill.go:1117-1145` still `make`s and zero-fills about 26 buffers per call; now paid per chunk: A-N01, C-N03. |
| N-16 | OPEN | `ensurePrefill` is still called inside `PrefillLast` (`metal/prefill.go:1091`); the first ≥64-token request pays the library compile: A-N01. |
| N-17 | CLOSED-VERIFIED | `gemm_w4f16` gone; `gemv_w4a8_sa_amax` documented in `metal/kernels.go:9-16`. |
| N-18 | CLOSED-VERIFIED | subsumed: fused kernel no longer the hd=128 path (R19); fused still serves hd < 128 (area A). |
| N-19 | CLOSED by measurement; unfused as observed | A: rope q+k, both norms and kv store total 3.19 ms at K=512 (0.2%) and 18.7 ms at K=3900 with the LM head (0.1%), `metal-prefill-decomp-2026-09-25.md:66,81`; F: `rope_f16` is still dispatched separately for Q and K (`metal/prefill.go:1283-1284`). Not proposed. |
| N-20 | CLOSED; replaced by C-P01 | The per-stage re-derivation is gone (`int4DirectBytesOnly`, `metal/gemma4_moe.go:292-293`, `metal/moe.go:600-602`); the retained f16 scale cache that replaced it duplicates v15-aliased scales (C-P01), so the finding's premise, an f32 heap copy, is stale under v15. |
| N-21 | CLOSED-VERIFIED | `metal/expertpool.go:213-218` uses `gpu.NewBufferLenOf`. |
| N-22 | Superseded by M-11 | Same status as M-11 (D). |
| N-23 | CLOSED in `moe.go`; OPEN and moot in `gemma4_moe.go` | `metal/gemma4_moe.go:606-612` keeps per-dense-layer buffers; M26 has no dense layers; the mixed dense+paged path has no fixture (D). |
| N-24 | f32 router weight NEGATIVE-CLOSED; single-thread route REVISIT | The f32 weight is deliberate (≤0.4 ms); `moe_route` on one GPU thread is D-P04. |
| N-25 | PARTIAL | Comment rewritten (`metal/backend.go:891-899`); still one command buffer per position (`:803-823`): A-P04, and F-N01 for the slot it overwrites. |
| N-26 | CLOSED-VERIFIED | `decoder/residency.go:60-69` states Metal's `ForwardBatch` is one command buffer, per-token only for paged MoE. |
| N-27 | NEGATIVE-CLOSED, unchanged | `metal/model.go:1955-1969` still copies; declined for the CUDA DMA-race reason (C, E). E-C01 is the one place the same hazard is latent. |
| N-28 | PARTIAL | Temperature-only requests are on-device since R7b (`gumbel.go`, `gumbel_sample.go`); filtered sampling is host-side (E-P04); the device sampler bypasses encode-ahead (C-P02). |
| N-29 | CLOSED-VERIFIED | `bytesToU32` at `metal/model.go:558`; `int4Concat` pre-sized (`:452-677`). |
| N-30 | PARTIAL; negative stands | Stale K≤1536 comment corrected (`metal/kernels.go:397-400`); `swiglu_quant` double evaluation is a recorded negative needing a device scratch buffer (audit `:1509-1520`); `rope2_kv` stays unwired (0.6%). |
| N-31 | OPEN (microseconds) | `WaitDone` still reads four GPU timestamps per production token (aikit `metal.go:740-767`); counted in C's ~10 command-buffer-level calls. |
| N-32 | PARTIAL | Comment fixed (`metal_vit.go:1149-1150`); the library-wide fast-math-off compile remains (`metal_vit.go:363`). |
| N-33 | CLOSED-VERIFIED | `metal/expertpool.go:66-77`. |
| N-34 | UNCHANGED, still correct on UMA | aikit `metal_copy.go`, `metal_upload_batch.go`: host memcpy, unused by goinfer (C). |
| N-35 | CLOSED-VERIFIED | `decoder/model.go:1504-1535` keyed set. |
| N-36 | CLOSED-VERIFIED | `decoder/residentneed.go:62`. |
| N-37 | OPEN as a caveat, mitigated | `gemv_w4a8_coal_bench_test.go` shape unchanged; R18 and R18b grade in sequence through `TestR18InSequence` and cite the Stage-B lesson (B). |
| N-38 | OPEN | first-call compile note; unchanged (and `ensurePrefill` is still lazy, N-16). |
| N-39 | PARTIAL | Comment fixed; the underlying gap stays: `decoder/model.go:1783` still requires `prefillFrom == 0` for the resident path with an adapter. |
| N-40 | PARTIAL | Code: the head is pinned int8 by decision (`decoder/weightmat.go:78-94`), head kernel at 161–163 GB/s, 90% of ceiling (B). Doc: `benchmarks.md:1026` still says "4-bit both sides" (F). The opt-in int4 head has no Metal kernel: B-N01, O3. |
| N-41 | PARTIAL, live in a new form | The ceiling moved 4096→32768 and an unpinned load defaults to 4096, so the 5689-position pin is now accepted; the planner still has no Metal ceiling and prices f32 KV: C-C01 (inflation, CPU fallback) and F-C02 (the exact kernel's `sc[4096]`). |

---

## 13. Not settleable statically

Each area report's section (f) lists its own; these are the ones that decide a Major or a REVISIT, with what settles
them. Everything here needs the Mac or a file the audited snapshot did not carry.

| Question | Decides | Settled by |
|---|---|---|
| GPU-idle share of the 0.45 ms non-GPU time per token | C-B01, C-P02, C-B03 | T1.3: `kernStart(t+1) − kernEnd(t)` |
| Post-R19 pass cost by depth at C = 16–64 | A-P03 | T1.1: `TestMC5_passCost`, no code |
| Whether `--kv i8` reaches `PrefillLast` in a served run; on-device behaviour of A-C01 | A-C01 ≡ F-C01 | T0.1's reproduction |
| Steel numerics at `startPos > 0`, with windows, at ragged M and K; chunk invariance on the steel build | F-G01, F-G02 | T1.5 and the F-G01 test |
| How often the fit guard auto-pins 4097–32768, or above 32768, on a 16 GB Mac | F-C02, C-C01 | The C-C01 unit test, then a load under memory pressure |
| Whether untouched KV pages of the extra slots count in the footprint at allocation | E-P09, C-C01 | T1.6 |
| Lowering of `simd_sum`; per-lane registers and occupancy of `sa_rows_acc` at R=4 and `attention_fa_blk<7>`; whether g8 spills | B-P04, B-P06, B-P07, B-P02 | A Metal compiler report or GPU capture on the Mac |
| Legacy attention cost at 512–1535 keys on the current build; blk cost below 2048 keys | B-P03 | T1.2 |
| fb2 against fb4 timings on the 7B; the dequant-versus-MMA split inside the fragment | E-P03 | Instantiate fb4 (the S0 logs are not in the snapshot) |
| Whether adjacent same-tile threadgroups share weight reads through L2/SLC on M1 Pro | E-P02 | The 2-row 7B gate\|up standalone dispatch |
| Whether `calibrateRows` is stable run to run, and its choice at B=2,3 on the 7B | E-P02 | Repeat `buildBatch` |
| Staging versus commit versus wait split on aliased M26 at N=8; N=64 and N=32 after the fork-collapse fix; M35 on Metal after S6 | D-P02, M-11 | T1.9 |
| Metal expert-major speedup and fidelity on a real MoE; router-flip rate; `testdata/mixtral-tiny`'s nE, k and shared-expert shape | D-G01, D-B02, D-P01 | The D-G01 fixture |
| 16-row against 64-row tile cost on an expert shape; per-dispatch encode cost | D-B02 | T1.12 |
| Real gpt-oss requantisation error; cost of an E2M1 unpack against `UNP8` | D-B03 | T1.13 |
| Recurrence share of a resident hybrid token; `delta_rule` gain from lane-per-Dk | D-B01, D-B05 | A resident-hybrid decomposition |
| Whether CI exports `GOINFER_METAL_MC3` and `GOINFER_HEAVY_TESTS`; what `cmd/gate gpu` names; where `TestBatchedVerifyKernelParity` went | E-G01, F-G02, F-G04 | Read the CI configuration and `cmd/gate` in the working repo |
| Whether `.giw` replacement is always temp+rename; what `prequant`'s default `-target` writes | F-N02, F-D02 | Read `internal/prequant` in the working repo |
| Whether any registry architecture has hd=128 with nH/nKV > 8 and a dense layer | F-C03 | Enumerate the registry |
| Whether any checkpoint beyond Qwen2.5-1.5B/7B overflows the f16 residual | A-C02 | The finite-check in T0.3 counts it in the field |
| Prompt-length distribution of real first turns | A-P02 | Not recorded anywhere; log it |
| MLX's per-chip commit bucket and `sdpa_vector` 2-pass switch for an M1 Pro; whether mlx-lm quantises `lm_head`; the same-session MLX ratio | §0, B-N01 | T1.14 (the first two affect only the MLX comparison, never a goinfer change) |
| GPU watchdog behaviour for a whole-prompt single command buffer | C-N02 | A long-prompt run |
| Files absent from the snapshot that the reports cite: `docs/completed/metal-verdict.md`, `scripts/autoresearch_rmsnorm_results.tsv`, `docs/tasks/task-concurrency-2026-09.md` and its log directories, `docs/task-mxfp4-gptoss.md`, the S0 logs | B-P09, M-09 loose ends, E-P03, D-B03 | Read them in the working repo before acting on those items |

<!-- citation-lint: allow-path visionmetal/threadgroup_budget_test.go aikit's own separate Go module (gpu/visionmetal), not pinned by goinfer, so no checked-out or module-cache root can verify it here; same reason as visionmetal/encoder.go in audit-metal-2026-09-12.md. -->
