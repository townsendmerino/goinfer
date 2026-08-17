package decoder

import (
	"context"
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
func (s *BlockSpec) Generate(prompt []int, opt BlockSpecOptions) ([]int, int, error) {
	return s.generate(prompt, opt, nil)
}

// generate is the loop. emit, when non-nil, receives each round's committed tokens as they are
// produced and returns false to stop (cancellation) — that is what lets GenerateStream forward
// tokens per round instead of at the end.
func (s *BlockSpec) generate(prompt []int, opt BlockSpecOptions, emit func([]int) bool) (out []int, rounds int, err error) {
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
	if emit != nil && !emit([]int{anchor}) {
		return out, rounds, nil
	}

	maskID := dw.MaskTokenID()
	var guard acceptanceGuard
	seamOff := false
	for opt.MaxTokens <= 0 || len(out) < opt.MaxTokens {
		if eos[anchor] {
			break
		}
		if guard.stopped {
			// The drafter is not paying for itself on this generation. Finish with plain
			// resident decoding rather than continuing to lose ~20% invisibly.
			//
			// DISARM THE CAPTURE SEAM FIRST. Leaving it armed makes every fallback token pay
			// five tap downloads for hidden states nothing will read — measured turning a
			// 0.98x generation into 0.87x, i.e. the guard made things WORSE than not guarding.
			if !seamOff {
				if e := host.SetBatchedCapture(nil); e != nil {
					return out, rounds, e
				}
				seamOff = true
			}
			one, e := host.PrefillLastNArgmax([][]float32{m.embedResident(anchor)}, pos)
			if e != nil {
				return out, rounds, e
			}
			pos++
			anchor = one[0]
			out = append(out, anchor)
			if emit != nil && !emit([]int{anchor}) {
				return out, rounds, nil
			}
			continue
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
		burst := make([]int, 0, accepted+1)
		burst = append(burst, drafted[:accepted]...)
		next := tgt[accepted] // the target's own token at the first disagreement
		burst = append(burst, next)
		out = append(out, burst...)
		if emit != nil && !emit(burst) {
			return out, rounds, nil
		}
		pos += 1 + accepted
		if e := fuse(capt, 1+accepted); e != nil {
			return out, rounds, e
		}
		anchor = next
		guard.observe(1 + accepted)
	}
	return out, rounds, nil
}

// GenerateStream is the serving-shaped entry point: greedy block-drafting speculation as a token
// channel, matching GenerateEagleSpeculative's signature so a server can swap one for the other.
//
// GREEDY ONLY, and the guards are the same ones the EAGLE path carries for the same reason. The
// verify compares the drafted token against the target's ARGMAX; a temperature, a logit processor
// or any history-dependent penalty makes "what the target would have produced" depend on state
// the batched verify does not have, so acceptance would no longer imply losslessness. Refusing is
// the honest answer — a caller that needs sampling should use Generate.
func (s *BlockSpec) GenerateStream(ctx context.Context, prompt []int, maxTokens int,
	sp SamplingParams) (<-chan int, *Generation, error) {
	if sp.Temperature != 0 || sp.LogitProcessor != nil {
		return nil, nil, fmt.Errorf("decoder.BlockSpec.GenerateStream: greedy only (no temperature/LogitProcessor)")
	}
	if sp.HistoryDependent() {
		return nil, nil, fmt.Errorf("decoder.BlockSpec.GenerateStream: repetition penalties / logit bias not supported in greedy speculative decoding; use Generate")
	}
	if !s.m.specRollbackSafe() {
		return nil, nil, fmt.Errorf("decoder.BlockSpec.GenerateStream: recurrent family or staged sliding-window unsupported (rollback cannot restore)")
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.BlockSpec.GenerateStream: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	go func() {
		defer close(out)
		// The loop emits in bursts (anchor plus accepted drafts), so tokens are forwarded as
		// each round commits rather than at the end — a server streams them straight through.
		emit := func(ids []int) bool {
			for _, id := range ids {
				select {
				case out <- id:
				case <-ctx.Done():
					return false
				}
			}
			return true
		}
		toks, rounds, err := s.generate(prompt, BlockSpecOptions{MaxTokens: maxTokens}, emit)
		stats.Rounds = rounds
		stats.Emitted = len(toks)
		// Accepted counts DRAFT tokens the target confirmed, excluding each round's own
		// correction token — the same convention SpecStats uses for the n-gram path, so the
		// two acceptance rates are comparable.
		if a := len(toks) - rounds - 1; a > 0 {
			stats.Accepted = a
		}
		if err != nil {
			g.err = err
		}
	}()
	return out, g, nil
}

// breakEvenTokensPerRound is the acceptance below which block drafting LOSES.
//
// A round costs draft + batched verify + the capture seam whatever it accepts; plain decoding
// costs one target forward per token. So the drafter pays only when it commits more tokens per
// round than the round costs in decode-equivalents. On the measured 4B/2070S pairing that is
// ~39 ms per round against an 11.1 ms decode — about 3.5.
//
// This is not a tuning knob, it is a break-even, and the two measurements that motivated it
// straddle it exactly: non-thinking Qwen3 accepts 5.76/round and runs 1.77x, while the SAME
// model in thinking mode accepts 3.00 and runs 0.98x. The margin below is deliberate: disabling
// a drafter that is merely breaking even costs nothing, while leaving one running that is losing
// costs 20% and is INVISIBLE, because block drafting is lossless — the output is identical
// either way, so an operator sees correct responses at reduced speed and has nothing to look at.
const breakEvenTokensPerRound = 3.8

// guardWindow is how many rounds to observe before judging. Long enough that a hard opening
// (the first block of a response is often unpredictable) does not disable a drafter that would
// have paid, short enough that a losing one is switched off within a fraction of a response.
const guardWindow = 12

// acceptanceGuard disables a drafter that is not paying for itself, per generation.
//
// It exists because the failure it catches is SILENT. A mis-paired drafter, a target in a mode
// the drafter was not trained for, or an out-of-domain workload all produce correct output at
// reduced speed — losslessness guarantees the tokens are right. Without this, `--drafter` on a
// thinking-mode Qwen3 serves at 0.83x and nothing anywhere says so.
type acceptanceGuard struct {
	rounds  int
	tokens  int
	stopped bool
}

// observe records a round and reports whether block drafting should continue.
func (g *acceptanceGuard) observe(committed int) bool {
	if g.stopped {
		return false
	}
	g.rounds++
	g.tokens += committed
	if g.rounds >= guardWindow {
		if float64(g.tokens)/float64(g.rounds) < breakEvenTokensPerRound {
			g.stopped = true
			return false
		}
		// Passed the check: keep going, and re-judge over the next window rather than
		// letting an early good stretch license an arbitrarily long bad one.
		g.rounds, g.tokens = 0, 0
	}
	return true
}
