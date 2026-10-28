package main

// `gate identity -record FILE` (TE6(b), docs/tasks/task-test-efficiency-2026-09.md): a family's validation may
// be inherited by identity. It writes PARITY_ROW lines for the decoder package's TestParityManifest_merge, the
// manifest's one writer, and never writes the manifest itself.
//
// A family is eligible only when identity is proof of the SAME numerics its T3 oracle validated:
//   - its row is `validated` under a T3 method (or an earlier identity-inherited one);
//   - the run used real checkpoints, not the tiny fixtures (a tiny fixture cannot reach the shape-dependent kernel
//     paths a released checkpoint does);
//   - the backend is the CPU, and the inner method is one the CPU reference carries (full-forward-oracle, weightDiff,
//     shared-path). A real-model-oracle row validated a GPU-resident path, which only that backend can inherit;
//   - this machine's arch is the arch the row was validated on (the CPU reference is bit-identical within an arch, not
//     across);
//   - <old-rev> is the row's validated_at, and <new-rev> is HEAD (the merge stamps HEAD);
//   - every cell of the family came back IDENTICAL, covering all three CPU quants (f32, int8int8, int4), so whichever
//     quant the oracle ran at is covered.
//
// The row keeps the original metrics, and its method carries the original method and the rev its oracle ran at:
// "identity-inherited (<method> @ <rev>)". The checkpoint that ran is named in the reference, and the merge is a
// deliberate step after checking it against the row's own reference.

import (
	"runtime"
)

var identityRecordQuants = []string{"f32", "int8int8", "int4"}

type inheritance struct {
	Family  string
	Row     string   // the PARITY_ROW line, when eligible
	Reasons []string // why not, when not
}

func identityRecordArch() string { return runtime.GOARCH }
