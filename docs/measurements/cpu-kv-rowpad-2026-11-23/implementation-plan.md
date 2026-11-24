# CPU KV row pad: the full implementation's plan, written 2026-11-23 20:31 PST

Go-ahead: Francis ("go with b if it's really good"), on the ADOPT read (README beside this file), relayed by the coordinator ~20:35 PST.
Code and tests first; the coordinator sees it before it is armed; its A/B is registered and run by day.

**Layout.** The pad moves from the prototype's side fields into the cache's own store: a dense global layer's `keys/vals` (f32) or
`keysH/valsH` (f16) hold rows `rowStride = kvDim + kvRowPadElems(kvDim, elemBytes, 4096)` elements apart; `c.stride[l]` records
that stride (it already means "the per-position stride", which `TruncateTo` and the share paths use). int8 and ring layers keep today's
layout (int8 rows are kvDim bytes, no aliasing class measured; rings are bounded by the window). The prototype's `keysP`/`keysPH`,
`setRowPad` side fields, `mustUnpadded` and `kvRowPadForTest` are removed; a test-only switch back to the unpadded layout stays for the A/B.

**Readers, each changed and each tested** (catalogued from the tree at `ff5bb32c2`):
- Decode: `attendToken`'s global f32 and f16 cases pass `rowStride` to `attendBatchedHeadsKV` (the parameter the prototype added);
  `attendQueryP` (`program_cpu.go`, the sink/tree path) and `attention.go`'s `Keys/Vals` reader index by the stride.
- Batched prefill and the batched walker: `program_cpu_n.go` (both `attendBatchedHeadsP` callers and the `c.Keys` reader at the
  per-row loop), `forwardn.go`'s append of K new rows (`Append` per row, so padded), the prefill row reservation (`reserveRows`,
  `appendHint`, `regrowLayer`: capacity in strided rows).
- Families with their own loops: `forward_gptoss.go`, `forward_qwen35.go`, `forward_gemma4.go`, `program_cpu_n_gemma4.go`,
  `program_cpu_n_qwen35.go`. Each either indexes by the stride or is declined the pad (gemma4's per-layer widths and KV sharing make
  its layers non-uniform; it stays unpadded, as the prototype's eligibility already does).
- Rewind and sharing: `TruncateTo` (already per `c.stride`), `prefix_share.go` and `session_share.go` (copy rows at the source's
  stride; a destination with a different stride re-packs), `kvsnapshot.go` (the snapshot format writes compact kvDim rows, so a
  snapshot taken padded restores unpadded and the reverse; `checkGlobalLen` checks against the stride in memory).
- Widened reads: `dequantGlobalLayer` (f16) and `batchReadLocal` widen strided rows into compact scratch.
- The GPU upload bridge: `LayerKV` (and `internal/wrappedupload` through `LayerKVForTest`) returns compact rows, the pad dropped.
- Accounting: `storedRows`, `cost_cpu.go`'s `kvRowBytes`, `kvBytesForRows` (the plan's price, as in the prototype), `testhooks.go`'s
  byte counter.

**Tests, red first:** per reader, bit-identity padded against unpadded through its real caller (decode, prefill, prefill-then-decode,
rewind then decode, prefix and session share, snapshot round trip both ways, the upload bridge's rows); the existing F1 exactness cells
on every fixture with the pad forced on (`kvRowPadAlign` lowered); the plan's price equal to the bytes a padded cache holds; and a census
that every function in `decoder/` indexing `keys`/`vals` by `kvDim` is on an allow list (so a new reader cannot skip the stride).

**Its A/B (registered before it runs):** Phi-3 f32 at 3,900 and 128, padded against unpadded on the 4-key and the 16-key kernel, which
settles the QK dispatch; the 0.5B as the control; the f16 read re-measured on the shipping kernel. And the default question (pad on for
every eligible CPU cache) goes to Francis with its memory: 128 B per token per layer, Phi-3 16 MiB at 4,096 positions.
