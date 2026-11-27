package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/blockdraft"
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/sampling"
)

type (
	DFlashContext            = blockdraft.Context
	DSparkDrafter            = blockdraft.DSpark
	DrafterRows              = blockdraft.DrafterRows
	DrafterLayerRows         = blockdraft.DrafterLayerRows
	StreamedDrafterWeights   = blockdraft.StreamedDrafterWeights
	StreamingDrafterHost     = blockdraft.StreamingDrafterHost
	ReleasableDrafterWeights = blockdraft.ReleasableDrafterWeights
)

var (
	ErrDrafterReleased = blockdraft.ErrReleased
	ErrDrafterStreamed = blockdraft.ErrStreamed
)

// DFlashDrafter is an imported z-lab DFlash block drafter.
type DFlashDrafter struct {
	*blockdraft.DFlash
}

func LoadDFlashDrafter(dir string) (*DFlashDrafter, error) {
	d, err := blockdraft.LoadDFlash(dir)
	if err != nil {
		return nil, err
	}
	return &DFlashDrafter{DFlash: d}, nil
}

func OpenDFlashDrafter(dir string) (*DFlashDrafter, error) {
	d, err := blockdraft.OpenDFlash(dir)
	if err != nil {
		return nil, err
	}
	return &DFlashDrafter{DFlash: d}, nil
}

func LoadDSparkDrafter(dir string) (*DSparkDrafter, error) {
	return blockdraft.LoadDSpark(dir)
}

func DrafterResidentBytesEstimate(dw BlockDrafterWeights) int64 {
	return blockdraft.EstimateResidentBytes(dw)
}

func DrafterKVBytesPerPosition(dw BlockDrafterWeights) int64 {
	return blockdraft.KVBytesPerPosition(dw)
}

var (
	_ StreamedDrafterWeights = (*DFlashDrafter)(nil)
	_ BlockDrafterWeights    = (*DSparkDrafter)(nil)
)

// EmbedBlock builds the trunk's block input for ids, using the drafter's own embedding when it has one and the
// target's otherwise. The fallback is not cosmetic: a drafter that ships embed_tokens was trained against those
// vectors, and the target's are a silent off-distribution input. The trunk still runs and output stays lossless, so
// the only symptom is acceptance landing below what the pairing is worth.
func (d *DFlashDrafter) EmbedBlock(m *Model, ids []int) [][]float32 {
	blockdraft.MustHoldWeights(&d.BlockTrunk) // a released embedding is zero-rowed, and would read as "borrows the target's"
	if !blockdraft.HasOwnEmbed(d.DFlash) {
		return m.DrafterEmbedBlock(ids)
	}
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, blockdraft.HiddenDim(&d.BlockTrunk))
		d.EmbedDraft(id, rows[i])
	}
	return rows
}

// DraftIDs turns the trunk's block output rows into TARGET token ids, using the
// drafter's own reduced-vocab head when present and the target's LM head otherwise.
// One call site for both drafter shapes, so a pairing cannot silently take the wrong
// output path.
func (d *DFlashDrafter) DraftIDs(m *Model, rows [][]float32) []int {
	blockdraft.MustHoldWeights(&d.BlockTrunk)
	out := make([]int, len(rows))
	for i, h := range rows {
		if d.HasOwnHead() {
			out[i] = d.DraftTokenID(m.be, h)
		} else {
			out[i] = sampling.Argmax(m.DrafterHeadLogits(h))
		}
	}
	return out
}

// DrafterHeadLogits applies the TARGET's LM head to one already-normed hidden row and
// returns next-token logits over the target vocabulary. It is the output end that a
// drafter shipping no `lm_head` of its own (DFlash) borrows from the target.
//
// It deliberately does NOT apply the target's final norm — the drafter has already
// applied its OWN `norm`, and running both would be a second normalization the reference
// does not do. That is the only difference from logitsFromHidden, and it is why this
// cannot just call it. The post-head transforms (Gemma softcap, Granite logit scale) ARE
// applied, mirroring the reference's compute_logits.
//
// Declared here rather than in model.go on purpose: model.go is in the parity manifest's core set, and adding a
// drafter-only accessor there would re-stale every family's deps_hash for a function no existing forward calls.
func (m *Model) DrafterHeadLogits(h []float32) []float32 {
	arch := m.w.Arch
	logits := make([]float32, arch.VocabSize)
	if arch.TiedLMHead {
		cpuengine.Matmul(m.be, &m.w.Embed, h, logits, 1)
	} else {
		cpuengine.Matmul(m.be, &m.w.LMHead, h, logits, 1)
	}
	if arch.FinalLogitSoftcap > 0 {
		cpuengine.SoftcapParallel(logits, float32(arch.FinalLogitSoftcap))
	}
	if arch.LogitScale != 0 && arch.LogitScale != 1 {
		inv := float32(1 / arch.LogitScale)
		for i := range logits {
			logits[i] *= inv
		}
	}
	return logits
}

// DrafterEmbedBlock fills the drafter's block input with the TARGET's token embeddings —
// the input end DFlash borrows. ids must be blockSize long (slot 0 the anchor, the rest
// the mask token).
func (m *Model) DrafterEmbedBlock(ids []int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		rows[i] = make([]float32, m.w.Arch.HiddenDim)
		m.embedToken(id, rows[i])
	}
	return rows
}
