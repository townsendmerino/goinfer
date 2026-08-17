package decoder

import (
	"errors"
	"fmt"
)

// errBlockSpecUnsupported is returned when the backend cannot host a block drafter, so a caller
// can fall back to plain generation instead of treating it as a failure.
var errBlockSpecUnsupported = errors.New("decoder: backend does not support block-drafting speculation")

// ErrBlockSpecUnsupported reports whether err is the decline above.
func ErrBlockSpecUnsupported(err error) bool { return errors.Is(err, errBlockSpecUnsupported) }

// The backend-side interfaces block-drafting speculation drives, mirroring ResidentForward.
//
// `decoder` owns the LOOP — draft, verify, accept, roll back, extend — because that logic is
// numerics-free and identical on every backend. What differs per backend is only how the
// drafter's trunk runs and how the target's hidden states come back, which is what these
// declare. The alternative, a loop written inside `cuda`, would have to be rewritten for Metal
// and could not be tested without a device.
//
// Both are OPTIONAL capabilities: a ResidentForward that does not implement them simply cannot
// host block drafting, and GenerateBlockSpec declines to plain generation rather than failing.

// ResidentBlockDrafter is a block drafter's trunk living on a backend's device, attached to that
// backend's resident target. Obtained from ResidentDrafterHost.AttachBlockDrafter.
type ResidentBlockDrafter interface {
	// FuseContext projects concatenated tap hidden states to the trunk's width and norms them.
	FuseContext(rows [][]float32) ([][]float32, error)
	// ExtendContext appends fused rows to the drafter's own K/V at the current context end.
	// Incremental by contract: rebuilding the whole context per block measured 2.4x the cost
	// at a 1024-token context (docs/spec/08), widening with length.
	ExtendContext(fused [][]float32) error
	// ContextLen is how many positions the drafter's context currently holds.
	ContextLen() int
	// TruncateContext drops context positions >= n — the rollback after a partial accept.
	TruncateContext(n int)
	// DraftBlock runs the trunk over one block, returning its output rows.
	DraftBlock(blockIn [][]float32) ([][]float32, error)
	// DraftTokens turns trunk rows into token ids using the TARGET's LM head — a block drafter
	// ships none of its own, which is why the pairing is fixed.
	DraftTokens(trunk [][]float32) ([]int, error)
}

// ResidentDrafterHost is a resident target that can host a block drafter and hand back the
// hidden states it needs. A backend implements this alongside ResidentForward.
type ResidentDrafterHost interface {
	// AttachBlockDrafter uploads a drafter's weights to this target's device. It must reject a
	// geometry mismatch rather than resize: a drafter reads the target's residual stream, so a
	// mismatch is the wrong PAIRING, and the failure mode is a drafter that runs and drafts noise.
	AttachBlockDrafter(w BlockDrafterWeights) (ResidentBlockDrafter, error)
	// PrefillLastNArgmax verifies M rows in one batched pass, returning each row's argmax token
	// — all the accept decision reads. Only the ARGMAX must match a sequential Forward, not the
	// logits bit-for-bit, which is what lets the LM head be batched (docs/spec/08).
	PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
	// SetBatchedCapture arms the hidden-state seam for the whole batch: the next
	// PrefillLastNArgmax records the residual at each named layer for ALL its rows. The
	// per-token seam costs a sync and a download per tap PER TOKEN; this pays one per tap for
	// the block (1.09 ms vs 2.79 ms at M=6, measured).
	SetBatchedCapture(taps []int) error
	// BatchedCapture returns the last batched forward's rows as [tap][M*hidden].
	BatchedCapture() [][]float32
}

// BlockSpecCapable reports whether this model can run block-drafting speculation: a resident
// target that implements ResidentDrafterHost. Checked before a drafter is loaded, so a caller
// can decline early rather than paying a load it cannot use.
func (m *Model) BlockSpecCapable() bool {
	if m.resident == nil {
		return false
	}
	_, ok := m.resident.(ResidentDrafterHost)
	return ok
}

// BlockSpecOptions tunes block-drafting speculation.
type BlockSpecOptions struct {
	// VerifyWidth is how many block positions the TARGET verifies per round (anchor plus
	// VerifyWidth-1 drafts). 0 selects the default below.
	//
	// It is NOT the drafter's trained block width, and that distinction is the single biggest
	// lever measured: the drafter drafts its full block either way, but verifying all 16
	// positions makes code a 0.89x LOSS, while verifying 7 makes it 1.60x. The tail positions
	// rarely land and cost full batched-verify price — positions 12-15 gain 0.09 accepted
	// tokens BETWEEN THEM while costing 9.4 ms of verify per round (docs/spec/08).
	VerifyWidth int
	// MaxTokens caps generation; 0 means unlimited (the caller stops on EOS).
	MaxTokens int
}

// defaultVerifyWidth is 8 — measured as the optimum for math (1.79x), within 2% of code's
// optimum of 7 (1.60x), and serving both pairings tested. Per-traffic-class tuning is worth
// a few percent (code 7, chat 4) and needs a router; 8 is the one number that works everywhere.
const defaultVerifyWidth = 8

// BlockSpec is an attached block drafter, ready to serve many generations.
//
// ATTACHING IS SEPARATE FROM GENERATING, and that split is not cosmetic. AttachBlockDrafter
// uploads the drafter's weights (~500 MB for the 4B pairing) to the device; doing it per request
// made the production path measure 0.17x — a 6x LOSS — while the loop itself was healthy and
// lossless at 5.76 tok/round. Acceptance looked fine and the wiring was throwing the speedup
// away. Attach once per process, generate per request.
type BlockSpec struct {
	m    *Model
	host ResidentDrafterHost
	rd   ResidentBlockDrafter
	dw   BlockDrafterWeights
	taps []int
}

// NewBlockSpec attaches a block drafter to this model's resident target. Returns a decline
// (ErrBlockSpecUnsupported) when the backend cannot host one, so a caller can try
// unconditionally and fall back to plain generation.
func (m *Model) NewBlockSpec(dw BlockDrafterWeights, taps []int) (*BlockSpec, error) {
	host, ok := m.resident.(ResidentDrafterHost)
	if !ok {
		return nil, errBlockSpecUnsupported
	}
	rd, err := host.AttachBlockDrafter(dw)
	if err != nil {
		return nil, err
	}
	return &BlockSpec{m: m, host: host, rd: rd, dw: dw, taps: taps}, nil
}

// Generate runs greedy generation with the attached drafter, returning the emitted token ids and
// the number of verify rounds.
//
// LOSSLESS BY CONSTRUCTION: every emitted token is one the TARGET's own argmax produced. The
// drafter only proposes; a proposal the target disagrees with is discarded along with everything
// after it, and the target's own token is emitted instead. So the output is token-identical to
// plain greedy whatever the drafter does — a bad drafter costs speed, never correctness. That is
// also why a broken drafter is INVISIBLE to a correctness test, and must be caught by watching
// acceptance instead (docs/spec/08 records five such errors).
//
// It declines rather than fails when the backend cannot host block drafting, so a caller can
// pass a drafter unconditionally and get plain generation where it is unsupported.
func (s *BlockSpec) Generate(prompt []int, opt BlockSpecOptions) (out []int, rounds int, err error) {
	m, host, rd, dw := s.m, s.host, s.rd, s.dw
	width := opt.VerifyWidth
	if width <= 0 {
		width = defaultVerifyWidth
	}
	if bs := dw.BlockSize(); width > bs {
		width = bs // never ask the target to verify positions the drafter did not draft
	}
	if width < 2 {
		return nil, 0, fmt.Errorf("decoder: block-spec verify width %d is too narrow", width)
	}
	if err := host.SetBatchedCapture(s.taps); err != nil {
		return nil, 0, err
	}
	defer func() { _ = host.SetBatchedCapture(nil) }()
	rd.TruncateContext(0) // fresh sequence: the previous generation's context must not leak in

	hidden := m.w.arch.HiddenDim
	eos := map[int]bool{}
	for _, e := range m.w.Cfg.EOSIDs() {
		eos[e] = true
	}
	// fuse folds a batched capture into the drafter's context: the taps for n rows become n
	// concatenated rows, projected and appended.
	fuse := func(capt [][]float32, n int) error {
		cat := make([][]float32, n)
		for i := 0; i < n; i++ {
			row := make([]float32, 0, len(capt)*hidden)
			for _, tp := range capt {
				row = append(row, tp[i*hidden:(i+1)*hidden]...)
			}
			cat[i] = row
		}
		fused, e := rd.FuseContext(cat)
		if e != nil {
			return e
		}
		return rd.ExtendContext(fused)
	}

	embs := make([][]float32, len(prompt))
	for i, id := range prompt {
		embs[i] = m.embedResident(id)
	}
	ids, err := host.PrefillLastNArgmax(embs, 0)
	if err != nil {
		return nil, 0, err
	}
	if err := fuse(host.BatchedCapture(), len(prompt)); err != nil {
		return nil, 0, err
	}
	pos := len(prompt)
	anchor := ids[len(ids)-1]
	out = append(out, anchor)

	maskID := dw.MaskTokenID()
	for opt.MaxTokens <= 0 || len(out) < opt.MaxTokens {
		if eos[anchor] {
			break
		}
		blockIn := make([][]float32, width)
		blockIn[0] = m.embedResident(anchor)
		for i := 1; i < width; i++ {
			blockIn[i] = m.embedResident(maskID)
		}
		trunk, e := rd.DraftBlock(blockIn)
		if e != nil {
			return out, rounds, e
		}
		drafted, e := rd.DraftTokens(trunk[1:])
		if e != nil {
			return out, rounds, e
		}
		vin := make([][]float32, 0, 1+len(drafted))
		vin = append(vin, m.embedResident(anchor))
		for _, id := range drafted {
			vin = append(vin, m.embedResident(id))
		}
		tgt, e := host.PrefillLastNArgmax(vin, pos)
		if e != nil {
			return out, rounds, e
		}
		capt := host.BatchedCapture()
		rounds++

		accepted := 0
		for i, d := range drafted {
			if tgt[i] != d {
				break
			}
			accepted = i + 1
		}
		for i := 0; i < accepted; i++ {
			out = append(out, drafted[i])
		}
		next := tgt[accepted] // the target's own token at the first disagreement
		out = append(out, next)
		pos += 1 + accepted
		if e := fuse(capt, 1+accepted); e != nil {
			return out, rounds, e
		}
		anchor = next
	}
	return out, rounds, nil
}
