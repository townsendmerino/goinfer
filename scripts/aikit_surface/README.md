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

## Proof, per move

- `testdata/kernel_fingerprints.json` byte-identical and `TestCompute_table` passing without `-update`, on each
  dispatch configuration the table holds;
- every module and tag builds (go vet: root untagged, `realckpt`, `goinfer_testhooks`; `backend/cuda` with
  `cuda goinfer_testhooks`; `backend/webgpu` with `gpu`; `backend/metal` with `metal` on darwin; the apps);
- ken and tack build against the aikit branch, and aikit's `go run -C tools ./retrievalmark --ken <ken>` passes.
