package moecache

import "slices"

// Pair identifies one (row, routing-rank) assignment to an expert, with optional routing weight.
type Pair struct {
	Row    int
	Rank   int
	Weight float32
}

// GroupByExpert buckets (row, rank) pairs from row-major hostIdx[rows*topK] by routed expert ID and returns the sorted
// distinct expert IDs alongside their pair slices in row-then-rank order.
func GroupByExpert(hostIdx []uint32, rows, topK int) ([]uint32, map[uint32][]Pair) {
	byExpert := make(map[uint32][]Pair)
	for m := range rows {
		base := m * topK
		for j := range topK {
			e := hostIdx[base+j]
			byExpert[e] = append(byExpert[e], Pair{Row: m, Rank: j})
		}
	}
	experts := make([]uint32, 0, len(byExpert))
	for e := range byExpert {
		experts = append(experts, e)
	}
	slices.Sort(experts)
	return experts, byExpert
}

// GroupWeightedByExpert buckets (row, rank, weight) assignments from row-major idx[rows*topK] and wgt[rows*topK] into a
// dense table indexed by expert ID in [0, numExperts). Out-of-range expert IDs are ignored.
func GroupWeightedByExpert(idx []uint32, wgt []float32, rows, topK, numExperts int) [][]Pair {
	byExpert := make([][]Pair, numExperts)
	for m := range rows {
		base := m * topK
		for j := range topK {
			eIdx := int(idx[base+j])
			if eIdx >= 0 && eIdx < numExperts {
				byExpert[eIdx] = append(byExpert[eIdx], Pair{Row: m, Rank: j, Weight: wgt[base+j]})
			}
		}
	}
	return byExpert
}

// SlotBatchEntries packs routed slot tables slots[row][:topK] into fixed-width entries of 2+maxPairs uint32s:
// [slot, count, pair_0, ..., pair_{maxPairs-1}], ordered by first-seen slot and chunked at maxPairs pairs per entry.
func SlotBatchEntries(slots [][]uint32, topK, maxPairs int) []uint32 {
	var order []uint32
	pairs := make(map[uint32][]uint32)
	for i, si := range slots {
		for j := range topK {
			s := si[j]
			if _, ok := pairs[s]; !ok {
				order = append(order, s)
			}
			pairs[s] = append(pairs[s], uint32(i*topK+j))
		}
	}
	var out []uint32
	stride := 2 + maxPairs
	for _, s := range order {
		for ps := pairs[s]; len(ps) > 0; {
			c := min(len(ps), maxPairs)
			e := make([]uint32, stride)
			e[0], e[1] = s, uint32(c)
			copy(e[2:], ps[:c])
			out = append(out, e...)
			ps = ps[c:]
		}
	}
	return out
}
