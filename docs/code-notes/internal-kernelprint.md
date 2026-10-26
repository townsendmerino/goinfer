# internal/kernelprint: design notes

The package's contract is in its `doc.go`. This file holds the design: why it has this shape, what it does not cover, and
where its evidence lives. Plan: `docs/tasks/task-subsystems-2026-10.md` §5.4 (VK1) and §5.8 (VK5, which mixes the
fingerprints into the parity key). Measurement record: `docs/measurements/vk1-2026-10.md`.

## What a fingerprint is for

`decoder/parity_manifest_test.go:freshDepsHash` mixes aikit's version string into every family's key, so every aikit
bump re-stales every family, including bumps that change no kernel goinfer calls (VK0: 46 commits and 1,876 family
re-stalings in 30 days). A fingerprint replaces the version string with what the version string stands for: the output
bits of the kernels goinfer calls, on inputs that reach their dispatch paths, per dispatch configuration. An aikit bump
that leaves every fingerprint alone then carries every validation over.

## Shape

- **One kernel, one fingerprint.** A key that names the kernels a family's operations use (VK4's program) re-stales
  only the families that use a changed kernel. A kernel that calls another (`QuantizeInt8` calls `QuantizeRowsInt8`)
  changes with it, which is correct: K2's "exactly that kernel" is about plants in a leaf.
- **Kernels are named as goinfer calls them**, not as aikit implements them: `linalg.WeightMat(int4).MatmulBTW4A8Into`
  over every int4 layout goinfer loads, not the eight row4 and split-half entry points behind it. The suite follows
  goinfer's calling conventions where they matter: `AttendTileFused` gets `MatmulBT` as its matmul, Workspaces get
  goinfer's two decode thresholds (`DefaultDecodeParallelThreshold`, `int4ParThreshold`) beside aikit's default and an
  always-parallel one, the weight and the Workspace carry the same activation group, `SoftmaxRowContractInto` also runs
  in place (as `internal/whisper` calls it), and the .giw reader's `WrapInt4Row4OnlyF16` is also called where it declines.
- **Inputs are architecture-independent bits.** They come from splitmix64 and exact float32 arithmetic only (sums, and
  products by powers of two inside explicit conversions), because the Go spec lets the compiler fuse `a*b+c` into an FMA,
  which gc does on arm64 and not on amd64; an explicit conversion rounds and prevents it. A fingerprint difference
  between architectures is then the kernels', never the inputs'.
- **Each kernel's seed is derived from its name**, so adding, removing or reordering kernels never moves another
  kernel's fingerprint.
- **Inputs include the edges a quantizer meets:** an all-zero row, a constant row, an outlier every 13 values, rows whose
  maximum is exactly 127 so x.5 values sit on int8 rounding ties, K below one group and ragged last groups (goinfer takes
  any `cols`), every binary16 and bfloat16 bit pattern for the widening conversions, and float32 probes at the rounding
  boundaries of both 16-bit formats.
- **Expected failures panic.** A GGUF or safetensors read that errors would otherwise hash an empty output and look
  stable; a panic makes K1 fail instead.

## K3: what "reaches every census path" means here

The census (`docs/hardware-coverage.json`, its `cpu-*` entries) names the dispatch predicates (`hasDotProd`, `hasAVX2`,
...). K3 parses aikit's `linalg` and `embed` for the configuration under test, finds every `if` whose condition names a
predicate (directly, or through a one-line `return` predicate such as `row4Usable`), and evaluates the condition with
the predicates known and every other operand unknown. A decided site needs its active side executed; an undecided one
needs both, which is how tails and thresholds behind a predicate (`!hasAVX2 || n < 16`) get demanded. Executed means a
block of the child run's coverage profile holding that statement ran.

Two kinds of exclusion, both checked where they can be:

- `uncalled`: a function goinfer's production code never reaches, with the exported entry points it is reached through.
  The test scans goinfer's non-test sources and fails if any names one of them.
- `unreachableSide`: one side of one site, under one verdict, that goinfer's calling conventions cannot produce (an
  attention row with no keys, a Q4_K row shorter than a super-block). These are claims about goinfer's callers that a
  scan cannot check; each says why. An entry whose file is built but whose `if` is gone fails as stale.

**Configurations.** K3 runs only in the configurations its exclusions were written against (`validated` in
`census_test.go`), and is VOID where a forcing tag forced nothing. An AVX-512 VNNI runner (GitHub's pool mixes them in)
reports that it was not validated there instead of failing intermittently.

## Known limits

- **embed's `hasAVX2` follows the CPU, not the tags.** `embed/gguf_dequant_q4k_amd64.go` declares its own
  `hasAVX2 = cpu.X86.HasAVX2`, which no `aikit_*` tag turns off. So on amd64 the pure-Go Q4_K and Q6_K GGUF dequantizers
  run on no machine here and in no CI cell, `aikit_noavx2` included. K3 reads embed's flag from the detected CPU so it
  does not demand a path the tag cannot reach; closing the gap is aikit's (a forcing hook for embed).
- **The towers are not covered.** aikit's `vision` and `audio` packages are not in the suite; Track AX moves them into
  goinfer, where they become goinfer code.
- **AVX-512 VNNI is not run here.** No machine here has it; the record names where it can run.
- **The split-half layout is fingerprinted though production does not build it** (`w4a8SplitHalfRepackEnabled` is set
  only by tests). It is in the suite because the code is shipped and one toggle away; a change there re-stales nothing
  goinfer runs today, a cost VK5's shadow mode will show if it matters.
