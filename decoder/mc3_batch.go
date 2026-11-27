package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/residency"

type (
	tokenCoalescer     = residency.TokenCoalescer
	residentBatcher    = residency.Batcher
	ResidentBatchStats = residency.BatchStats
	batchReq           = residency.BatchReq
	chunkReq           = residency.ChunkReq
)

const (
	batchStragglerWait = residency.BatchStragglerWait
	mc3HoldSendGrace   = residency.MC3HoldSendGrace
	prefillTailMin     = residency.PrefillTailMin
)

// ResidentBatchStats is a snapshot of the batcher's counters; zero when the batcher is not enabled.
func (m *Model) ResidentBatchStats() ResidentBatchStats {
	if m.batcher == nil {
		return ResidentBatchStats{}
	}
	return m.batcher.Stats()
}

// EnableResidentConcurrency lets up to n generations use this model's resident at once.
func (m *Model) EnableResidentConcurrency(n int) int {
	lo, hi, ok := stepBatchRange(m.resident)
	if !ok || n < 2 {
		return 1
	}
	slots := m.resHost().SlotCount()
	if hi < 2 || slots < 2 {
		return 1
	}
	n = min(n, slots)
	stepSeqs := func(seqs []ResidentBatchSeq) ([]ResidentBatchOut, error) {
		sr, err := stepOn(m.resident, stepBatch{Pass: passSeqs, Seqs: seqs})
		return sr.Batch, err
	}
	var stepRagged func([]ResidentRaggedSeq) ([]ResidentRaggedOut, error)
	if canStep(m.resident, stepBatch{Pass: passRagged, Ragged: []ResidentRaggedSeq{{Embs: stepProbe}}}) {
		stepRagged = func(seqs []ResidentRaggedSeq) ([]ResidentRaggedOut, error) {
			sr, err := stepOn(m.resident, stepBatch{Pass: passRagged, Ragged: seqs})
			return sr.Ragged, err
		}
	}
	m.batcher = residency.NewBatcher(m.resident, n, slots, lo, hi, stepSeqs, stepRagged)
	return n
}

// ResidentConcurrency is how many generations may use the resident at once.
func (m *Model) ResidentConcurrency() int {
	if m.batcher == nil {
		return 1
	}
	return m.batcher.MaxHolders()
}

// SetSpecAdaptive turns the "speculate when alone, batch under load" candidate on or off.
func (m *Model) SetSpecAdaptive(on bool) { m.specAdaptive = on }

// ResidentBatchCapable reports whether EnableResidentConcurrency could enable anything on this model.
func (m *Model) ResidentBatchCapable() bool {
	_, hi, ok := stepBatchRange(m.resident)
	if !ok || m.resHost().SlotCount() < 2 {
		return false
	}
	return hi >= 2
}
