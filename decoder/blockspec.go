package decoder

import (
	"context"
	"errors"
	"fmt"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"sync/atomic"
)

// errBlockSpecUnsupported is returned when the backend cannot host a block drafter, so a caller
// can fall back to plain generation instead of treating it as a failure.
var errBlockSpecUnsupported = errors.New("decoder: backend does not support block-drafting speculation")

// ErrBlockSpecUnsupported reports whether err is the decline above.
func ErrBlockSpecUnsupported(err error) bool { return errors.Is(err, errBlockSpecUnsupported) }

// errBlockSpecResidentBusy is returned when another generation already holds the model's single shared resident KV. It is returned before
// any device write, so a caller's fallback to plain Generate (internal/serveapp/openai.go) is exact: Generate makes its own claim attempt
// and, finding it held, drops to the staged CPU path itself.
var errBlockSpecResidentBusy = errors.New("decoder: resident KV is busy with another generation")

// ErrBlockSpecResidentBusy reports whether err is the concurrent-claim decline above.
func ErrBlockSpecResidentBusy(err error) bool { return errors.Is(err, errBlockSpecResidentBusy) }

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

// RecurrentSpecMaxRows is the widest verify a RecurrentSpecAvailable arena is sized for: a DFlash block.
const RecurrentSpecMaxRows = 16

// RecurrentSpecArenaBytes is the device memory a RecurrentSpecAvailable resident keeps for verify passes of up to rows rows on a model of
// arch a: two banks (a verify writes one while the previous verify's committed rows fold from the other) of, per Gated-DeltaNet layer and
// row, the conv input and output, the normed q and k and two gates per value head, plus one fold output, f32, rounded up to the 2 MiB
// cuMemAlloc gives a buffer that size. 0 for a model the path refuses: no recurrent
// state, a recurrent kind other than Gated DeltaNet, heads wider than the scan twins' 128, or a sliding window.
func RecurrentSpecArenaBytes(a *darch.Architecture, rows int) int64 {
	if a == nil || a.Qwen35 == nil || rows < 1 || a.SlidingWindow > 0 || a.Qwen35.KeyHeadDim > 128 || a.Qwen35.ValueHeadDim > 128 {
		return 0
	}
	nDN := 0
	for l := range a.NumLayers {
		for _, o := range a.LayerOps(l, false) {
			switch {
			case o.Kind == opDeltaNet:
				nDN++
			case o.Kind.Recurrent():
				return 0
			}
		}
	}
	if nDN == 0 {
		return 0
	}
	g := a.Qwen35
	keyDim, valueDim := g.NumKeyHeads*g.KeyHeadDim, g.NumValueHeads*g.ValueHeadDim
	convDim := 2*keyDim + valueDim
	floats := 2*int64(nDN)*int64(rows)*int64(2*convDim+2*keyDim+2*g.NumValueHeads) + int64(rows)*int64(valueDim)
	const align = 2 << 20
	return (floats*4 + align - 1) / align * align
}

// recurrentSpecReserve is what a block drafter's attach adds to the companion reserve on top of its own bytes: the fold path's arena on
// a recurrent target (RecurrentSpecArenaBytes), 0 on any other.
func recurrentSpecReserve(a *darch.Architecture, drafter bool) int64 {
	if !drafter {
		return 0
	}
	return RecurrentSpecArenaBytes(a, RecurrentSpecMaxRows)
}

// BlockSpecCapable reports whether this model can run block-drafting speculation: a resident
// target that implements ResidentDrafterHost. Checked before a drafter is loaded, so a caller
// can decline early rather than paying a load it cannot use.
func (m *Model) BlockSpecCapable() bool {
	return m.resident != nil && m.resRuntime().DrafterHostAvailable()
}

// BlockSpecOptions tunes block-drafting speculation.
type BlockSpecOptions struct {
	// VerifyWidth is how many block positions the TARGET verifies per round (anchor plus VerifyWidth-1 drafts). 0 selects the default below.
	//
	// It is not the drafter's trained block width, and the distinction is the biggest lever: the drafter drafts its full block either way, but
	// the tail positions rarely land and cost full batched-verify price, so verifying the full width can turn a win into a loss.
	// docs/code-notes/decoder.md#BlockSpecOptions.VerifyWidth has the measurement.
	VerifyWidth int
	// MaxTokens caps generation; 0 means unlimited (the caller stops on EOS).
	MaxTokens int

	// StopIDs are the caller's extra stop tokens: SamplingParams.StopIDs, which for a served request carries the chat template's stops on top
	// of the model's own. The loop's stop set must agree with Model.isStop, which every other speculative loop consults (blockSpecStopSet);
	// otherwise a stop token is emitted as content and the lossless contract is broken by the stop set rather than the verify.
	StopIDs []int

	// OnRound, if non-nil, is called after each completed round with the verify width that round used and the tokens it committed
	// (accepted drafts plus the target's own token). It is the per-round seam for the adaptive controller, which cannot be judged on
	// end-to-end speed alone (a cumulative average lags a regime change mid-generation), and for `serve`'s per-request accept-rate and
	// tok/verify metadata.
	OnRound func(width, committed int)

	// scope is the request's KV sharing scope (KVScope), from GenerateStream's ctx; the zero value is the local one.
	scope KVScope
}

// blockSpecFuseRows is the most rows one FuseContext call takes. A drafter's fusion scratch grows with the rows of one call, so a
// long prompt's seed in one call can exhaust the device beside the target.
const blockSpecFuseRows = 512

// defaultVerifyWidth is 8, the one number that serves both measured pairings (the optimum for math, within 2% of code's optimum of 7).
// Per-traffic-class tuning is worth a few percent and needs a router; docs/code-notes/decoder.md#defaultVerifyWidth has the figures.
const defaultVerifyWidth = 8

// BlockSpec is an attached block drafter, ready to serve many generations.
//
// Attaching is separate from generating, and that split is not cosmetic: AttachBlockDrafter uploads the drafter's weights to the device,
// and doing it per request costs more than the speedup earns (a several-fold loss) while the loop stays lossless and acceptance looks
// fine. Attach once per process, generate per request.
type BlockSpec struct {
	m *Model
	// host is the resident itself, never one of its subsystems: step and SubsystemsOf resolve a resident, and a subsystem handed to
	// either has no capability (blockSpecHost).
	host  ResidentDrafterHost
	rd    ResidentBlockDrafter
	dw    BlockDrafterWeights
	taps  []int
	width int // the verify width SetVerifyWidth set; 0 is defaultVerifyWidth
}

// VerifyWidthInUse is the verify width GenerateStream, and Generate without its own VerifyWidth, run at: SetVerifyWidth's value
// after the default and the caps apply.
func (s *BlockSpec) VerifyWidthInUse() int {
	return s.verifyWidth(BlockSpecOptions{})
}

// SetVerifyWidth sets the verify width GenerateStream, and Generate without its own VerifyWidth, run at (BlockSpecOptions.VerifyWidth's
// meaning: 0 is the default, and the drafter's block caps it). It refuses a width under 2 and a width GenerateStream would refuse
// (specRollbackSafeVia): on a recurrent target, one its resident's fold path cannot verify.
// Set it once, before the BlockSpec serves; it is not safe beside a running generation.
func (s *BlockSpec) SetVerifyWidth(w int) error {
	if w != 0 && w < 2 {
		return fmt.Errorf("decoder: block-spec verify width %d: want 0 (the default) or at least 2", w)
	}
	v := w
	if v == 0 {
		v = defaultVerifyWidth // the width 0 selects, not the one set before (verifyWidth falls back to s.width)
	}
	// Only the drafter's block caps the request here. verifyWidth's other cap, to the fold path's prepared rows, must not hide a
	// width past them: that width is refused, as GenerateStream would refuse it.
	v = min(v, s.dw.BlockSize())
	if !s.m.specRollbackSafeVia(s.recurrentFold(v), v) {
		return fmt.Errorf("decoder: block-spec verify width %d (%d after the drafter's block of %d) cannot roll back on this model (a recurrent model past its resident's fold rows, or a staged sliding window)",
			w, v, s.dw.BlockSize())
	}
	s.width = w
	return nil
}

// NewBlockSpec attaches a block drafter to this model's resident target. Returns a decline
// (ErrBlockSpecUnsupported) when the backend cannot host one, so a caller can try
// unconditionally and fall back to plain generation.
func (m *Model) NewBlockSpec(dw BlockDrafterWeights, taps []int) (*BlockSpec, error) {
	_, done, err := m.enterCall(context.Background())
	if err != nil {
		return nil, err
	}
	defer done()
	out, err := m.newBlockSpec(dw, taps)
	return out, m.closeErr(context.Background(), err)
}

// newBlockSpec is NewBlockSpec without the lifecycle guard (lifecycle.go).
func (m *Model) newBlockSpec(dw BlockDrafterWeights, taps []int) (*BlockSpec, error) {
	// Checked BEFORE attaching: block drafting verifies every round through the resident's batched
	// path, so a resident whose decode and verify can disagree must never get a drafter uploaded to
	// it. The spike is fixed at load, so refusing here covers every later GenerateStream too.
	if err := m.SpecDecodeConflict(); err != nil {
		return nil, err
	}
	host, rt, ok := m.blockSpecHost()
	if !ok {
		return nil, errBlockSpecUnsupported
	}
	if sw, ok := dw.(StreamedDrafterWeights); ok && sw.Streamed() && !rt.CanStreamDrafterRows() {
		return nil, fmt.Errorf("decoder: this resident cannot take a drafter opened to stream its matrices (OpenDFlashDrafter); load it with LoadDFlashDrafter")
	}
	// Before the attach, so a refused model has no drafter uploaded to its device.
	if err := m.prepareRecurrentSpec(host, dw); err != nil {
		return nil, err
	}
	rd, err := rt.AttachBlockDrafter(dw)
	if err != nil {
		return nil, err
	}
	return &BlockSpec{m: m, host: host, rd: rd, dw: dw, taps: taps}, nil
}

// blockSpecHost is m's resident as a drafter host with its runtime subsystem; ok is false where the resident cannot host a block
// drafter. The host is the resident, not rt: a backend's subsystems are not ResidentSubsystemProviders, so SubsystemsOf(rt) is the
// no-op subsystems.
func (m *Model) blockSpecHost() (host ResidentDrafterHost, rt ResidentRuntimeSubsystem, ok bool) {
	rt = m.resRuntime()
	if !rt.DrafterHostAvailable() {
		return nil, nil, false
	}
	host, ok = drafterHostOf(m.resident)
	return host, rt, ok
}

// prepareRecurrentSpec readies a recurrent target's fold path for dw's block, or says why block drafting would decode from
// over-advanced state; nil for a model with no recurrent state.
func (m *Model) prepareRecurrentSpec(host ResidentDrafterHost, dw BlockDrafterWeights) error {
	if !m.hasRecurrentState() {
		return nil
	}
	rt := SubsystemsOf(host).Runtime
	if !rt.RecurrentSpecAvailable() {
		return fmt.Errorf("decoder: this model has recurrent state and its resident cannot verify without writing it (no RecurrentSpecAvailable)")
	}
	// The architecture first, so a model the path refuses has no arena allocated either; the arena's price is 0 exactly there.
	if RecurrentSpecArenaBytes(m.w.Arch, 1) == 0 {
		return fmt.Errorf("decoder: the fold path does not serve this model (a recurrent layer that is not Gated DeltaNet, heads past 128, or a sliding window)")
	}
	rows := min(dw.BlockSize(), RecurrentSpecMaxRows) // the widest verify this drafter's loop asks the fold path for
	if rows < 2 {
		return fmt.Errorf("decoder: a drafter block of %d positions verifies no draft", dw.BlockSize())
	}
	if err := rt.PrepareRecurrentSpec(rows); err != nil {
		return fmt.Errorf("decoder: this model has recurrent state and its resident's fold path declined: %w", err)
	}
	if !m.specRollbackSafeVia(rt, rows) {
		return fmt.Errorf("decoder: the resident's fold path holds %d rows, and this drafter verifies up to %d", rt.RecurrentSpecRows(), rows)
	}
	return nil
}

// recurrentFold is the resident's fold path for verifies of width rows on a recurrent model, or nil: on a model with no
// recurrent state (its verifies need none), and on one whose resident cannot take that width (specRollbackSafeVia).
func (s *BlockSpec) recurrentFold(width int) ResidentRuntimeSubsystem {
	if !s.m.hasRecurrentState() {
		return nil
	}
	rt := SubsystemsOf(s.host).Runtime
	if !rt.RecurrentSpecAvailable() || !s.m.specRollbackSafeVia(rt, width) {
		return nil
	}
	return rt
}

// verifyWidth is the verify width a generation with opt runs at: opt.VerifyWidth, else SetVerifyWidth's, else defaultVerifyWidth,
// clamped to the drafter's block and, on a recurrent target, to the rows the fold path is prepared for.
func (s *BlockSpec) verifyWidth(opt BlockSpecOptions) int {
	width := opt.VerifyWidth
	if width <= 0 {
		width = s.width // SetVerifyWidth's, for Generate as for GenerateStream
	}
	if width <= 0 {
		width = defaultVerifyWidth
	}
	if bs := s.dw.BlockSize(); width > bs {
		width = bs // never ask the target to verify positions the drafter did not draft
	}
	if s.m.hasRecurrentState() && width > RecurrentSpecMaxRows {
		width = RecurrentSpecMaxRows // the fold path's arena is prepared for at most this many rows (prepareRecurrentSpec)
	}
	return width
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

// seedAnchor runs the prompt suffix embs from pos through the host and returns the seed's id, the argmax of its last row. It asks for
// the id alone when the backend can (SeedArgmaxAvailable), which skips an M x vocab logits buffer, copy and host argmax.
func (s *BlockSpec) seedAnchor(embs [][]float32, pos int) (int, error) {
	if canStep(s.host, stepBatch{Pass: passRows, Emb: stepProbe, Out: outArgmax}) {
		sr, err := stepOn(s.host, stepBatch{Pass: passRows, Emb: embs, Pos: pos, Out: outArgmax})
		return sr.ID, err
	}
	ids, err := s.host.PrefillLastNArgmax(embs, pos)
	if err != nil {
		return 0, err
	}
	return ids[len(ids)-1], nil
}

// generate is the loop. emit, when non-nil, receives each round's committed tokens as they are
// produced and returns false to stop (cancellation) — that is what lets GenerateStream forward
// tokens per round instead of at the end.
func (s *BlockSpec) generate(prompt []int, opt BlockSpecOptions, emit func([]int) bool) (out []int, rounds int, err error) {
	m, host, rd, dw := s.m, s.host, s.rd, s.dw
	// Defense in depth, not load-bearing today: this loop takes no M=1 decode step (it seeds through a batched pass and then only verifies),
	// so with the multi-row verify lane off its verify rows are exact whether or not this scope is held. It is here so a future edit that
	// adds an M=1 step cannot silently mix attention trees. Counted, so GenerateStream's hold nests.
	defer m.enterExactAttention()()
	width := s.verifyWidth(opt)
	if width < 2 {
		return nil, 0, fmt.Errorf("decoder: block-spec verify width %d is too narrow", width)
	}
	// The token budget is plain decode's (generateInto): at most the context left after the prompt. Unclamped, a turn that reaches
	// the cap emits one more token than plain decode (a round's own last token, which has no room to be forwarded), and the guard's
	// fallback steps past the cap and fails where plain decode finishes.
	if ctxCap := m.ResidentContextCap(); ctxCap > 0 {
		switch left := ctxCap - len(prompt); {
		case left == 0:
			return nil, 0, nil // the prompt fills the context: plain decode emits nothing either
		case left > 0 && (opt.MaxTokens <= 0 || opt.MaxTokens > left):
			opt.MaxTokens = left
		}
	}
	// A recurrent target verifies through the fold path, which never writes the committed state; without it a verify would
	// advance the state by every row, rejected or not.
	fold := s.recurrentFold(width)
	if m.hasRecurrentState() && fold == nil {
		return nil, 0, fmt.Errorf("decoder: block-spec verify width %d on a recurrent model needs the resident's fold path (RecurrentSpecAvailable)", width)
	}
	if fold != nil {
		// Every exit leaves nothing marked: the committing one flushes below, and an early one (a stopped stream, a device or
		// drafter error) drops what it marked, so the next turn cannot fold rows from a sequence that was never committed.
		defer fold.DropFold()
	}
	// Claim the single shared resident KV before any device write: this path drives the same positional cache as Model.Generate and the
	// n-gram/speculative paths. A loser returns before touching state, so the caller's fallback to plain Generate is exact. Gated on
	// m.resident != nil, mirroring model.go's useGPU check: a host without a resident keeps its KV in its own CPU cache and must not
	// contend for resBusy or forget a commit it never wrote.
	var reuseFrom int
	if m.resident != nil {
		// tryClaimResident, not a bare CAS: it also refuses while a batched generation holds a batch place, which never
		// sets resBusy.
		if !m.tryClaimResident() {
			return nil, 0, errBlockSpecResidentBusy
		}
		defer atomic.StoreInt32(&m.resBusy, 0)
		// Compute reuse before forgetting: forgetting clears the very state residentReuseLen reads. BlockSpec never binds an adapter (nil lora,
		// matching residentCommitIDs' own nil below).
		// Binds the resident KV slot (clearing resDrafterSynced on a switch). Off: the drafter keeps its own context, which a
		// copy of another sequence's rows would not bring with it.
		sc := opt.scope
		sc.Off = true
		reuseFrom = m.resHost().AcquireIn(prompt, nil, 0, nil, sc).Reuse
		// A matching bound sequence is NOT enough on its own: a plain Generate or n-gram-speculative turn
		// can commit one without ever touching THIS drafter's own context, so the token
		// prefix can match while rd's state does not reflect it at all. Only trust reuseFrom
		// when the last resident write was this same *BlockSpec's own completed generate().
		if m.resDrafterSynced != s {
			reuseFrom = 0
		}
		// Forget FIRST — from here until the generation completes (or this call returns early)
		// the resident KV is mid-write, so the next turn must cold-prefill rather than trust it
		// (decoder/resident_reuse.go).
		m.resHost().ForgetIDs()
	}
	if err := host.SetBatchedCapture(s.taps); err != nil {
		return nil, 0, err
	}
	defer func() { _ = host.SetBatchedCapture(nil) }()
	// Keep the drafter's own context for the reused prefix instead of wiping to 0: its positions correspond 1:1 with the target's (each
	// prompt token appends exactly one fused row via fuse, below), so TruncateContext(reuseFrom) is the drafter-side twin of the target's
	// resident-KV reuse. reuseFrom is 0 (a full wipe) whenever nothing is safely reusable.
	rd.TruncateContext(reuseFrom)

	hidden := m.w.Arch.HiddenDim
	eos := blockSpecStopSet(m, opt)
	// fuse folds a batched capture into the drafter's context: the taps for n rows become n concatenated rows, projected and
	// appended, blockSpecFuseRows at a time. The fusion is per row and ExtendContext appends, so the pieces give the context one
	// call would.
	fuse := func(capt [][]float32, n int) error {
		for lo := 0; lo < n; lo += blockSpecFuseRows {
			hi := min(lo+blockSpecFuseRows, n)
			cat := make([][]float32, 0, hi-lo)
			for i := lo; i < hi; i++ {
				row := make([]float32, 0, len(capt)*hidden)
				for _, tp := range capt {
					row = append(row, tp[i*hidden:(i+1)*hidden]...)
				}
				cat = append(cat, row)
			}
			fused, e := rd.FuseContext(cat)
			if e != nil {
				return e
			}
			if e := rd.ExtendContext(fused); e != nil {
				return e
			}
		}
		return nil
	}

	suffix := prompt[reuseFrom:]
	embs := make([][]float32, len(suffix))
	for i, id := range suffix {
		embs[i] = m.embedResident(id)
	}
	anchor, err := s.seedAnchor(embs, reuseFrom)
	if err != nil {
		return nil, 0, err
	}
	if err := fuse(host.BatchedCapture(), len(suffix)); err != nil {
		return nil, 0, err
	}
	pos := len(prompt)
	if eos[anchor] {
		return out, rounds, nil // Generate emits nothing when the first token is a stop
	}
	out = append(out, anchor)
	if emit != nil && !emit([]int{anchor}) {
		return out, rounds, nil
	}

	maskID := dw.MaskTokenID()
	var guard acceptanceGuard
	seamOff := false
	var fallbackRow [1][]float32 // the plain-decode fallback's one-row batch, reused across its tokens
	for opt.MaxTokens <= 0 || len(out) < opt.MaxTokens {
		if eos[anchor] {
			break
		}
		if guard.stopped {
			// The drafter is not paying for itself on this generation: finish with plain resident decoding. Disarm the capture seam first: left
			// armed, every fallback token pays tap downloads for hidden states nothing will read, and the guard then costs more than it saves.
			if !seamOff {
				if fold != nil {
					if e := fold.FlushFold(); e != nil { // the plain steps read the committed state
						return out, rounds, e
					}
				}
				if e := host.SetBatchedCapture(nil); e != nil {
					return out, rounds, e
				}
				seamOff = true
			}
			// Use the fast greedy step where the backend has one (argmax reduced on-device with a 4-byte readback, as Model.Generate does).
			// Falling back through PrefillLastNArgmax(M=1) downloads the full logit row per token, so the guard would turn a loss into the same
			// loss instead of into plain-decode speed.
			var e error
			if m.canDecode(outArgmax) {
				var sr stepResult
				fallbackRow[0] = m.embedResident(anchor)
				sr, e = m.step(stepBatch{Pass: passToken, Emb: fallbackRow[:], Pos: pos, Out: outArgmax})
				anchor = sr.ID
			} else {
				var one []int
				one, e = host.PrefillLastNArgmax([][]float32{m.embedResident(anchor)}, pos)
				if e == nil {
					anchor = one[0]
				}
			}
			if e != nil {
				return out, rounds, e
			}
			pos++
			// The stop token is not emitted here either: the loop-top check breaks on it only after this append, so without this check the fallback
			// emits one token past where plain decoding stops.
			if eos[anchor] {
				break
			}
			out = append(out, anchor)
			if emit != nil && !emit([]int{anchor}) {
				return out, rounds, nil
			}
			continue
		}
		// The round's width is clamped by what is left of both budgets (blockSpecRoundWidth): the token budget is tested per round while a round
		// commits up to width tokens, and an unclamped verify near the end of the context window makes checkCap refuse the whole round.
		width := blockSpecRoundWidth(width, opt.MaxTokens, len(out), pos, m.ResidentContextCap())
		if width < 1 {
			break // no room in either budget: finish cleanly rather than erroring
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
		var tgt []int
		if fold != nil {
			tgt, e = fold.VerifyLastNArgmax(vin, pos)
		} else {
			tgt, e = host.PrefillLastNArgmax(vin, pos)
		}
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
		// Truncate before EOS inside the burst. A round commits several tokens at once, so a stop token can land in the middle of one;
		// appending the whole burst emits content after it, which plain decoding never does. The stop token itself is excluded, because Generate
		// breaks on it without emitting (model.go, isStop): matching that exactly is what makes the two paths token-identical.
		stop := false
		for i, id := range burst {
			if eos[id] {
				burst, stop = burst[:i], true
				break
			}
		}
		// The rows this round commits: the anchor and the drafts it emits. A stop inside the burst emits fewer than accepted, and
		// the position, the drafter's context and a recurrent state must all end at the last emitted token, which is what the
		// commit below records.
		rows := 1 + min(accepted, len(burst))
		if fold != nil { // they fold at the next verify, or at a flush below
			if e := fold.FoldVerified(rows); e != nil {
				return out, rounds, e
			}
		}
		out = append(out, burst...)
		if emit != nil && !stop && !emit(burst) {
			return out, rounds, nil
		}
		pos += rows
		if e := fuse(capt, rows); e != nil {
			return out, rounds, e
		}
		if stop {
			if emit != nil && len(burst) > 0 {
				emit(burst)
			}
			break
		}
		anchor = next
		// `width` here is the clamped width this round actually verified, which is what the telemetry must count: drafted = width-1 (the anchor
		// is not a draft), evaluated = the positions tested.
		if opt.OnRound != nil {
			opt.OnRound(width, 1+accepted)
		}
		guard.observe(1 + accepted)
	}
	// The rows the last round marked fold before the trailing forward and the commit read the state.
	if fold != nil {
		if err := fold.FlushFold(); err != nil {
			return out, rounds, err
		}
	}
	// The only exit that commits, mirroring generateInto's rule that a generation which ran to completion is the only place the resident
	// cache's contents are recorded (model.go). Every early return above (a device error, or the caller's emit stopping consumption) leaves
	// the bound slot forgotten, so the next turn cold-prefills rather than trusting a state this function cannot vouch for. resDrafterSynced = s marks this
	// BlockSpec's own drafter context as in sync with the commit below, so its next generate() can reuse both the target's resident KV and
	// the drafter's context; any other writer's commit (plain Generate, n-gram) clears it via residentForgetIDs, so a drafter turn never
	// trusts a context it never built.
	if m.resident != nil {
		// The trailing token. The seed and every round end on the target's own output: emitted, but not yet forwarded,
		// because it is the next round's anchor. So at this exit the cache holds prompt+out less its last token, and
		// committing prompt+out claimed a position it never received — a rejected draft's K/V, or nothing — which the
		// next turn would reuse as it stood (TestBlockSpecGenerate_commitsOnlyWrittenPositions). Plain decode forwards
		// every token it emits, so forward it here: the M=1 step the guard's fallback takes, its capture folded into
		// the drafter's context while the seam is armed, as a round's rows are. With no room in the context, or on a
		// forward error, commit only what was written.
		keep := len(out)
		if len(prompt)+len(out) > pos {
			keep = max(0, pos-len(prompt))
			ctxCap := m.ResidentContextCap()
			if len(prompt)+len(out) == pos+1 && (ctxCap <= 0 || pos < ctxCap) {
				emb := m.embedResident(out[len(out)-1])
				var e error
				if !seamOff {
					if _, e = host.PrefillLastNArgmax([][]float32{emb}, pos); e == nil {
						e = fuse(host.BatchedCapture(), 1)
					}
				} else if m.canDecode(outArgmax) {
					_, e = m.step(stepBatch{Pass: passToken, Emb: [][]float32{emb}, Pos: pos, Out: outArgmax})
				} else {
					_, e = host.PrefillLastNArgmax([][]float32{emb}, pos)
				}
				if e == nil {
					keep = len(out)
				}
			}
		}
		m.resHost().CommitIDs(prompt, out[:keep], nil, 0) // BlockSpec never binds an adapter
		m.resDrafterSynced = s
	}
	return out, rounds, nil
}

// blockSpecStopSet is this loop's stop predicate, and it must agree with Model.isStop, the predicate plain decoding and every other
// speculative loop use: m.eosIDs (config.json's eos_token_id merged with generation_config.json's additions by resolveEOSIDs) plus the
// caller's SamplingParams.StopIDs (for a served request, the chat template's stops). Building it from Cfg.EOSIDs() alone emits a stop
// token as content and breaks "lossless by construction" through the stop set rather than the verify.
func blockSpecStopSet(m *Model, opt BlockSpecOptions) map[int]bool {
	eos := make(map[int]bool, len(m.eosIDs)+len(opt.StopIDs))
	for _, e := range m.eosIDs {
		eos[e] = true
	}
	for _, e := range opt.StopIDs {
		eos[e] = true
	}
	return eos
}

// blockSpecRoundWidth clamps a round's verify width to what is left of both budgets. It returns < 1 when neither has room, which the
// caller treats as a clean finish. Two limits, one clamp: the loop tests its token budget once per round while a round commits up to
// `width` tokens, so max_tokens could be exceeded; and verifying `width` rows at `pos` past the context window makes the backend's
// checkCap refuse the whole round, where plain Generate clamps and a max-length turn finishes cleanly with "length".
func blockSpecRoundWidth(width, maxTokens, emitted, pos, ctxCap int) int {
	if maxTokens > 0 {
		if left := maxTokens - emitted; left < width {
			width = left
		}
	}
	if ctxCap > 0 && pos+width > ctxCap {
		width = ctxCap - pos
	}
	return width
}

// GenerateStream is the serving-shaped entry point: greedy block-drafting speculation as a token
// channel, the same shape as the n-gram entry points so a server can swap one for the other.
//
// GREEDY ONLY, and the guards are the same ones the draft-model path (GenerateSpeculative) carries. The
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
	if w := s.verifyWidth(BlockSpecOptions{}); !s.m.specRollbackSafeVia(s.recurrentFold(w), w) {
		return nil, nil, fmt.Errorf("decoder.BlockSpec.GenerateStream: recurrent family without the resident's fold path, or staged sliding-window, unsupported (rollback cannot restore)")
	}
	if len(prompt) == 0 {
		return nil, nil, fmt.Errorf("decoder.BlockSpec.GenerateStream: empty prompt")
	}
	out := make(chan int)
	stats := &SpecStats{}
	g := &Generation{Spec: stats}
	leaveExact := s.m.enterExactAttention() // decode and verify must share one attention tree
	s.m.runGen(ctx, out, g, leaveExact, func(ctx context.Context) {
		defer leaveExact()
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
		// Drafted/Evaluated come from the per-round hook because only the loop knows the clamped width each round used; computing them here from
		// len(toks) would assume every round ran at the configured width.
		var drafted, evaluated int
		opt := BlockSpecOptions{
			MaxTokens: maxTokens,
			StopIDs:   sp.StopIDs,
			scope:     kvScopeOf(ctx),
			OnRound: func(width, committed int) {
				drafted += width - 1 // the anchor is the target's own token, not a draft
				evaluated += committed
			},
		}
		toks, rounds, err := s.generate(prompt, opt, emit)
		stats.Rounds = rounds
		stats.Emitted = len(toks)
		stats.Drafted = drafted
		stats.Evaluated = evaluated
		// Accepted counts DRAFT tokens the target confirmed, excluding each round's own
		// correction token — the same convention SpecStats uses for the n-gram path, so the
		// two acceptance rates are comparable.
		if a := len(toks) - rounds - 1; a > 0 {
			stats.Accepted = a
		}
		if err != nil {
			g.err = err
		} else if ctx.Err() != nil {
			// A cancelled generation reports ctx.Err(), so the caller does not read a truncated stream as a clean finish.
			g.err = ctx.Err()
		}
	})
	return out, g, nil
}

// breakEvenTokensPerRound is the acceptance below which the guard disables block drafting.
//
// A round costs draft + batched verify + the capture seam whatever it accepts, against one target forward per token, so true break-even is
// higher than this. The threshold sits BELOW break-even on purpose: acceptance over the first few rounds is not acceptance over the
// generation (math opens slowly and is a win end to end), and a false negative (disabling a paying workload) costs far more than a false
// positive (a few unprofitable rounds before tripping). The guard should fire only when a workload is clearly losing. Do not raise it toward
// break-even: that disabled a paying workload. docs/code-notes/decoder.md#breakEvenTokensPerRound has the measurements.
const breakEvenTokensPerRound = 2.5

// guardWindow is how many rounds to observe before judging. Six: a shorter window falsely trips workloads that open slowly (a response's
// opening is not uniformly predictable, and the whole-generation average hides a slow start), while a longer one spends more rounds on a
// drafter that is not paying. The window should err long, because a false negative costs far more than a false positive. Do not shorten
// it without re-running the whole-generation comparison: docs/code-notes/decoder.md#guardWindow has the table and the reverted three-round
// attempt.
const guardWindow = 6

// blockSpecGuardTripForTest makes the guard stop every generation at its window (TripBlockSpecGuardForTest). Set by tests only.
var blockSpecGuardTripForTest bool

// acceptanceGuard disables a drafter that is not paying for itself, per generation.
//
// It exists because the failure it catches is silent: a mis-paired drafter, a target in a mode the drafter was not trained for, or an
// out-of-domain workload all produce correct output at reduced speed, since losslessness guarantees the tokens are right. Without it
// `--drafter` on such a workload serves slower than plain decoding and nothing says so.
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
	// Cumulative, not per-window: resetting the counters after a passing window gives a losing workload a fresh budget every time. The
	// running average trips a workload as soon as its evidence says so, while a slow-starting profitable one recovers as later rounds pull
	// the average up instead of being judged on a snapshot.
	if g.rounds >= guardWindow {
		if blockSpecGuardTripForTest || float64(g.tokens)/float64(g.rounds) < breakEvenTokensPerRound {
			g.stopped = true
			return false
		}
	}
	return true
}
