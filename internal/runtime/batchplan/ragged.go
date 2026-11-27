package batchplan

import (
	"fmt"

	"github.com/townsendmerino/goinfer/decoder/resident"
)

// RaggedOrder validates and orders StepRagged sequences so at most one prefill chunk appears first (carrying no draw,
// ending within PromptLen, and setting Last iff it reaches PromptLen), followed by all decode and verify sequences in
// caller order.
func RaggedOrder(prefix string, seqs []resident.RaggedSeq) ([]int, error) {
	order := make([]int, 0, len(seqs))
	for i, s := range seqs {
		if s.Prefill == nil {
			continue
		}
		end := s.Pos + len(s.Embs)
		switch {
		case len(order) > 0:
			return nil, fmt.Errorf("%s: sequences %d and %d are both prefill chunks; a step takes one", prefix, order[0], i)
		case s.Draw != nil:
			return nil, fmt.Errorf("%s: sequence %d is a prefill chunk with a draw", prefix, i)
		case s.Prefill.PromptLen < end:
			return nil, fmt.Errorf("%s: sequence %d's chunk ends at %d, past its prompt's %d", prefix, i, end, s.Prefill.PromptLen)
		case s.Prefill.Last != (end == s.Prefill.PromptLen):
			return nil, fmt.Errorf("%s: sequence %d's chunk ends at %d of a %d-row prompt with Last %v", prefix, i, end, s.Prefill.PromptLen, s.Prefill.Last)
		}
		order = append(order, i)
	}
	for i, s := range seqs {
		if s.Prefill == nil {
			order = append(order, i)
		}
	}
	return order, nil
}

// CheckDistinctSlots verifies that each sequence in order names a distinct in-range KV slot in [0, numSlots) and
// supplies at least one row.
func CheckDistinctSlots(prefix string, numSlots int, seqs []resident.RaggedSeq, order []int) error {
	seen := make(map[int]bool, len(order))
	for _, i := range order {
		s := seqs[i]
		switch {
		case s.Slot < 0 || s.Slot >= numSlots:
			return fmt.Errorf("%s: sequence %d names KV slot %d of %d", prefix, i, s.Slot, numSlots)
		case seen[s.Slot]:
			return fmt.Errorf("%s: KV slot %d is named by two sequences", prefix, s.Slot)
		case len(s.Embs) == 0:
			return fmt.Errorf("%s: sequence %d has no rows", prefix, i)
		}
		seen[s.Slot] = true
	}
	return nil
}
