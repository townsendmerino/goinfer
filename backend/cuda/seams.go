//go:build cuda

package cuda

// testSeams holds cudaResident's test plants and fault injections, embedded so each reads as a field of the resident. Only a
// test sets one, on the resident it holds; each is zero in production.
type testSeams struct {
	// costPlantForTest plants defects for the cost counters' checks to catch: 1 launches one kernel a token twice, 2 adds a host
	// wait after the token.
	costPlantForTest int
	// raggedPlantForTest plants a defect in StepRagged's rows, for its identity test's controls: 1 binds a sequence's second row
	// to the next sequence's slot (a row reading another sequence's tables), 2 gives each row of a sequence its first row's
	// position (a verify row run as a decode row), 3 runs a prefill chunk on the exact kernels (a chunk out of its pass's class),
	// 4 reads the ragged rows' tables with a stride one entry short, so slot s's row reads its table s entries back: a row's table
	// offset off by blocks, inside the tables the device holds. 0 in production. Set by tests only.
	raggedPlantForTest int
	// prefillDeclinePassForTest is the 1-based pass of a chunked prefill that declines without running, set by tests only.
	prefillDeclinePassForTest int
	// dnSpecPlantForTest plants a defect for the speculative verify's identity gate to catch. Set by tests only.
	dnSpecPlantForTest int
	// loraAllocFailAfterForTest, when > 0, makes the Nth LoRA buffer allocation of a bind fail as a device OOM (test seam; zero in
	// production). It counts down as buffers are allocated.
	loraAllocFailAfterForTest int
	// loraPanicAfterForTest, when > 0, makes the Nth LoRA buffer allocation of a bind panic with a value that is not a device allocation
	// failure (test seam; zero in production).
	loraPanicAfterForTest int
	// deepDefectForTest plants a defect for the DeepStack gate to catch: 1 the sets not added, 2 each set one layer late, 3
	// the sets added to the text rows too.
	deepDefectForTest int
}
