- **The Metal ledger cell reads its baseline before the load** (MM3c, tests only). Metal counts small buffers by the
  shared chunk it packs them into. The cell read its baseline after a chunk was already open, so 3,692 B of llama-tiny's
  small buffers were charged by the ledger but never counted by the device, and the cell failed. The ledger charges what
  Metal allocated (`allocatedSize` equals the length on every buffer). The band is unchanged, and the cell passes on an
  M1 Pro. A diagnostic probe of small-buffer counting and a per-buffer padding log come with it.
