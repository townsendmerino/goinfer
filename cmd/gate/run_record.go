package main

import (
	"fmt"
	"io"
	"os"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

// replayParity prints a recorded parity sweep's report as the sweep printed it: the runner's prelude, each cell's lines
// from its graded stream, gradeParity on the recorded inputs, and the runner's postlude. It writes no parity record;
// the sweep that ran did, and the postlude says where.
func replayParity(w io.Writer, path string) int {
	cfg, res, cells, rec, err := loadRunRecordFull(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate: %v\n", err)
		return 2
	}
	in, err := gatecore.UnmarshalPayload[parityInputs](rec.Parity)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate: %s: %v\n", path, err)
		return 2
	}
	if rec.Gate != "parity" || in == nil {
		fmt.Fprintf(os.Stderr, "gate: %s is not a parity sweep's record\n", path)
		return 2
	}
	fmt.Fprint(w, rec.Prelude)
	for _, c := range cells {
		fmt.Fprintf(w, "-- running %s ...\n", c.Cell.Name)
		writeParityCell(w, c)
	}
	blockers, _, _ := gradeParity(w, res, cfg.Cells, cells, *in)
	fmt.Fprint(w, rec.Postlude)
	if blockers > 0 {
		return 1
	}
	return 0
}
