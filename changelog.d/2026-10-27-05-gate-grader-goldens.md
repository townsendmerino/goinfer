- **`cmd/gate`'s graders are pinned by goldens on recorded inputs** (SB12, step 1). `TestGraderGoldens` regrades
  `testdata/gate/goldens/`: recorded `go test -json` streams and the committed gpu, parity, quick-full and `.giw`
  byte-identity records, with every git lookup the grader makes recorded in the case, so it runs on a shallow checkout.
  `TestGraderCensus_everySubcommand` holds the census of where each gate decides
  (`docs/measurements/sb12-2026-10.md`) to the subcommands `gate` dispatches. No gate's verdict changes. Found: `gate
  heavy -stream` has never counted a replayed stream (it reports RED with nothing counted).
