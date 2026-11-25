# aikit linalg surface rewrite

goinfer's side of aikit's linalg surface rework (the notes task `task-aikit-linalg-surface-2026-11`, rows LS2 and LS4).
The aikit changes wait on aikit's `linalg-surface-2026-11` branch; goinfer's bump is this rewrite, re-run on `main`'s
head when that branch lands (LS8), not a kept branch.

    python3 scripts/aikit_surface/aikit_surface_rewrite.py scripts/aikit_surface/*.rules -- .

Each `*.rules` file is one move: a `package <name> <import path>` line and one `gofmt -r` rule per moved name. A second
run changes nothing. Comments and strings are not rewritten: kernelprint's labels keep their `linalg.` names on purpose
(a label seeds its kernel's inputs and keys the fingerprint table). The script lists every comment that still names a
moved symbol; those are edited by hand.

## What is done by hand, per move

- `elem.rules` (the elementwise kernels): the two comments the script lists, in `backend/cuda/audio_encoder_test.go`
  and `backend/cuda/kernels/gelu_quant_test.go`.
- `attn.rules` (the attention kernels): nothing.
- `bitvec.rules` (sign-bit packing and Hamming): goinfer calls none of them; ken's one use (`internal/search/dense_store.go`)
  is rewritten by the same script, run in ken.

- LS6, typed `Kind` (aikit `linalg.Kind`, a string type whose `String()` keeps today's names): `kind_string_fix.py`
  adds `.String()` wherever goinfer uses `Kind()` as a plain string (a return, an argument, a map key, a comparison with a
  string variable): 12 files. Two test interfaces in `backend/cuda` (`reale2e_test.go`, `realforward_test.go`) declare
  `Kind() linalg.Kind` by hand.

- LS6 step 3 (narrowed on 2026-11-24): `policy.rules` renames the S-05 fold's A/B toggle to its test-hook names in
  `decoder/r9_s05_fold_ab_test.go`, and its two comments by hand. The other setters are unchanged.

- LS6 steps 1 and 2 run as one command:
  `python3 scripts/aikit_surface/ls6_fix.py scripts/aikit_surface/int4_layout.rules scripts/aikit_surface/batch.rules -- .`

- LS6 step 1 (one int4 constructor with a layout): `int4_layout.rules`. The rules map `Int4Row4Usable`, `WrapInt4Row4OnlyF16` and
  `RepackInt4Row4InPlaceF16` (and the split-half pair) one to one; the script turns the assignment and return forms of
  `WrapInt4F16` into `WrapInt4(..., LayoutCanonical)`, and lists what is left. 11 files. By hand: `decoder/weightmat.go`'s
  `WrapInt4F16` passed straight as an argument (a local first); and `WrapInt4Row4F16`, which has no replacement, in
  `decoder/weightmat_testhelpers_test.go` (its helper deleted) and its one caller, `decoder/w4a8_row4_mmapvheap_probe_test.go`,
  whose heap copy is now a row4-only `WrapInt4(LayoutRow4)` (the timed call reads only the row4 bytes either way).
  Not done here: `internal/diag/kernelprint/census_test.go`'s exclusion keys name the old aikit functions; K3 is already
  red against the integration branch (LS3's flag renames) and its keys change with goinfer's bump.

- LS6 step 2 (the batch call through `WeightMat`): `batch.rules` turns `W8A8Op`/`W4A8Op` into `BatchOp`, a W8A8 op
  literal built from one weight (`W8A8Op{BQ: wmInt8(w), Scales: wmScales(w), ...}`) into `BatchOp{W: w, Dst: d}`, and
  both batch calls into `MatmulBTBatchInto`. 21 files. By hand: `wmW4A8Op` (`decoder/weightmat.go`) builds
  `BatchOp{W: w, Dst: dst}`; backend/webgpu's `MatmulW8A8Batch` and `MatmulW4A8Batch` read the bytes through `op.W.Int8()`
  and `op.W.Int4F16()` (residency still keyed by the same backing pointer); kernelprint's two batch kernels wrap their
  quantized bytes in a `WeightMat` (labels and draw order unchanged); four tests that built ops from raw slices
  (`decoder/backend_w4a8_batch_test.go`, `decoder/parthreshold_test.go`, `backend/webgpu/internal/kernels/batch_test.go`,
  and `backend/webgpu/backend_w4a8_batch_test.go`, which builds only under `gpu goinfer_testhooks`);
  and the comments that named the old calls.

- LS3's follow-up (aikit's CPU flags in `internal/cpufeat`, its kernels in `linalg/attn`, `linalg/elem` and
  `linalg/bitvec`): `python3 scripts/aikit_surface/ls3_census_fix.py <aikit checkout>` re-keys
  `docs/hardware-coverage.json`'s cpu predicates and K3's exclusion keys (`internal/diag/kernelprint/census_test.go`)
  to the files and conditions as they now are, reading each file's location from the aikit checkout; then
  `python3 scripts/aikit_surface/ls3_census_code.py` makes the code edits that go with it (K3's packages, its `Has*`
  predicate map and a `cpufeat.HasX` selector in its evaluator, the moved packages' qualifiers; hwcensus reading the
  go.work aikit and matching `Has<X>`; prezerolint reading the kernel packages' output contracts, with exported
  sentinels and a floor of 50 writers), each an exact passage that fails by name if main changed it. By hand: the
  JSON's `files` lists (informational). hwcensus reads go.mod's pin when no go.work is in effect, so it is red on the
  released aikit until the bump, and green against the branch.

Order on main's head at LS8: `aikit_surface_rewrite.py` (the four rules files), `kind_string_fix.py` per module and tag,
`ls6_fix.py` (both rules files), `ls3_census_fix.py`, `ls3_census_code.py`, the hand edits above, then
`scripts/refresh_parity_hashes.sh`.

- LS2 part 2 (the int4 repack helpers and `WeightMat.Int4`/`Int4Row4` unexported): goinfer calls none of the
  helpers. By hand: three tests took a `*WeightMat` through an interface literal declaring `Int4()`
  (`decoder/int4mix_test.go`, and `backend/cuda`'s `reale2e_test.go` and `realforward_test.go`); they declare
  `Int4F16()` and copy the binary16 scales as stored (the f32 round trip they did was exact).

## Proof, per move

- `testdata/kernel_fingerprints.json` byte-identical and `TestCompute_table` passing without `-update`, on each
  dispatch configuration the table holds;
- every module and tag builds (go vet: root untagged, `realckpt`, `goinfer_testhooks`; `backend/cuda` with
  `cuda goinfer_testhooks`; `backend/webgpu` with `gpu`; `backend/metal` with `metal` on darwin; the apps);
- ken and tack build against the aikit branch, and aikit's `go run -C tools ./retrievalmark --ken <ken>` passes.
