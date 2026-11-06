# N-40: CUDA LoRA leaked buffers on a mid-upload OOM (nobara-sb01, 2026-11-06)

`SetAdapter` (`backend/cuda/lora.go`) built an adapter's device buffers in a local slice. A device allocation that fails mid-upload panics (`gpu.NewBufferLenOfOn`, by contract); `runJob` turned the panic into an error, but the buffers allocated for earlier layers and projections were
owned by nothing and stayed on the device ledger until the device closed. A `b` upload failing after its `a` succeeded orphaned `a` the same way.

- **Red first** (`red.log`): `TestLoRAUploadOOM_returnsEveryBuffer` (llama-tiny resident on the card; a test seam, `loraAllocFailAfterForTest`, fails the Nth LoRA buffer allocation of a bind, N from 1 to the whole bind, 56 buffers) asserts the device allocation ledger (`Device.LedgerLen`) returns to its pre-bind count, no adapter or cache stays bound, and a later bind allocates the same 56 buffers. Before the change every N from 2 on leaves buffers on the ledger (55 failing points; the leak per point is the buffers already allocated).
- **Change:** the upload loop is `uploadLoraLayers`; each projection's struct is stored in the caller's slice before its buffers are allocated, a `device allocation failed` panic is returned as an error (any other panic propagates), and `SetAdapter` releases the whole slice on any error. The rank-out-of-range and `setupErr` paths release the same way.
- **Green** (`green.log`, `green-final.log`): the new test, `TestLoRAAdapterCacheCUDA` and `TestLoRAResidentParityCUDA` pass (0.3 s, 1.9 s, 0.2 s); `touched.py` 13 checks and `cmd/lint` 7 of 7 pass (`touched.log`, `lint.log`).
- **Plants** (neither committed): no release on the error path -> red (`plant-no-release.log`, 55 failing points); the struct stored after both uploads (an orphaned `a`) -> red (`plant-orphan-a.log`).
- The test sits in `lora_cache_test.go` because the decoder-import ratchet (`internal/layout`) refuses a new test file that imports decoder; the red and plant logs were taken with the same test body in its first file.
- **Not in this PR:** N-43 (`resident.go` mid-loop return on a buffer error) is a separate audit item.
