package decoder

import (
	"context"
	"errors"
	"fmt"
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

// ResidentBlockDrafter is a block drafter's trunk living on a backend's device, attached to that
// backend's resident target. Obtained from ResidentDrafterHost.AttachBlockDrafter.
type ResidentBlockDrafter interface {
	// FuseContext projects concatenated tap hidden states to the trunk's width and norms them.
	FuseContext(rows [][]float32) ([][]float32, error)
	// ExtendContext appends fused rows to the drafter's own K/V at the current context end. Incremental by contract: rebuilding the whole
	// context per block costs more, and the gap widens with length (docs/spec/08).
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
	// SetBatchedCapture arms the hidden-state seam for the whole batch: the next PrefillLastNArgmax records the residual at each named layer
	// for ALL its rows. The per-token seam costs a sync and a download per tap per token; this pays one per tap for the block.
	SetBatchedCapture(taps []int) error
	// BatchedCapture returns the last batched forward's rows as [tap][M*hidden].
	BatchedCapture() [][]float32
}

// ResidentSeedArgmax is an OPTIONAL narrowing of PrefillLastNArgmax for the prompt seed, the one place the loop asks for M rows of argmax
// and reads exactly one of them. Without it the seed materialises the batched logits (M x vocab: gigabytes on a large-vocab target at
// prompt lengths over ~1k) and runs the head over every row to obtain one token id. The capture the seed needs comes from the layer loop,
// not the head, so heading one row loses nothing.
//
// Optional rather than part of ResidentDrafterHost, following ResidentCapped and ResidentGreedy: a backend without it keeps working
// through the wide path.
type ResidentSeedArgmax interface {
	// PrefillSeedArgmax runs the same batched forward and returns only the LAST row's argmax.
	// The batched capture seam must be armed and filled exactly as PrefillLastNArgmax fills it.
	PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error)
}

// ResidentRecurrentSpec is the OPTIONAL capability that lets a resident with recurrent state verify speculatively (SP2,
// docs/tasks/task-forward-architecture-2026-10.md §6a, "SP2 on nobara"). Without it, BlockSpec refuses every recurrent model,
// because PrefillLastNArgmax advances the state by every row it verifies, rejected or not.
type ResidentRecurrentSpec interface {
	// RecurrentSpecRows is the widest verify the path holds: 0 when this load cannot take it (a recurrent layer it does not
	// serve, or no room for its rows).
	RecurrentSpecRows() int
	// VerifyLastNArgmax is PrefillLastNArgmax that leaves every recurrent layer's committed state and conv window untouched.
	// Each row starts from the row above it, so the rows are a chain. The rows wait for FoldVerified; another verify replaces
	// them.
	VerifyLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
	// FoldVerified folds the last verify's first n rows into the committed state, leaving it as a prefill of those rows would.
	FoldVerified(n int) error
}

// RecurrentSpecMaxRows is the widest verify a ResidentRecurrentSpec arena is sized for: a DFlash block.
const RecurrentSpecMaxRows = 16

// RecurrentSpecArenaBytes is the device memory a ResidentRecurrentSpec resident keeps for verify passes of up to rows rows on a model of
// arch a: per Gated-DeltaNet layer and row, the conv input and output, the normed q and k and two gates per value head, plus the fold's
// recurrence output once, f32, rounded up to the 2 MiB cuMemAlloc gives a buffer that size. 0 for a model with no recurrent state or with
// a recurrent kind the path does not serve.
func RecurrentSpecArenaBytes(a *Architecture, rows int) int64 {
	if a == nil || a.qwen35 == nil || rows < 1 {
		return 0
	}
	nDN := 0
	for l := range a.NumLayers {
		for _, o := range a.layerOps(l, false) {
			switch {
			case o.kind == opDeltaNet:
				nDN++
			case o.kind.recurrent():
				return 0
			}
		}
	}
	if nDN == 0 {
		return 0
	}
	g := a.qwen35
	keyDim, valueDim := g.NumKeyHeads*g.KeyHeadDim, g.NumValueHeads*g.ValueHeadDim
	convDim := 2*keyDim + valueDim
	floats := int64(nDN)*int64(rows)*int64(2*convDim+2*keyDim+2*g.NumValueHeads) + int64(rows)*int64(valueDim)
	const align = 2 << 20
	return (floats*4 + align - 1) / align * align
}

// recurrentSpecReserve is what a block drafter's attach adds to the companion reserve on top of its own bytes: the fold path's arena on
// a recurrent target (RecurrentSpecArenaBytes), 0 on any other.
func recurrentSpecReserve(a *Architecture, drafter bool) int64 {
	if !drafter {
		return 0
	}
	return RecurrentSpecArenaBytes(a, RecurrentSpecMaxRows)
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

// defaultVerifyWidth is 8, the one number that serves both measured pairings (the optimum for math, within 2% of code's optimum of 7).
// Per-traffic-class tuning is worth a few percent and needs a router; docs/code-notes/decoder.md#defaultVerifyWidth has the figures.
const defaultVerifyWidth = 8

// BlockSpec is an attached block drafter, ready to serve many generations.
//
// Attaching is separate from generating, and that split is not cosmetic: AttachBlockDrafter uploads the drafter's weights to the device,
// and doing it per request costs more than the speedup earns (a several-fold loss) while the loop stays lossless and acceptance looks
// fine. Attach once per process, generate per request.
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
	host, ok := m.resident.(ResidentDrafterHost)
	if !ok {
		return nil, errBlockSpecUnsupported
	}
	rd, err := host.AttachBlockDrafter(dw)
	if err != nil {
		return nil, err
	}
	s := &BlockSpec{m: m, host: host, rd: rd, dw: dw, taps: taps}
	if m.hasRecurrentState() && s.recurrentFold(2) == nil {
		return nil, fmt.Errorf("decoder: this model has recurrent state and its resident cannot verify without writing it " +
			"(no ResidentRecurrentSpec path for every recurrent layer); block drafting would decode from over-advanced state")
	}
	return s, nil
}

// recurrentFold is the resident's fold path for verifies of width rows on a recurrent model, or nil: on a model with no
// recurrent state (its verifies need none), and on one whose resident cannot take that width (specRollbackSafeVia).
func (s *BlockSpec) recurrentFold(width int) ResidentRecurrentSpec {
	if !s.m.hasRecurrentState() {
		return nil
	}
	rs, _ := s.host.(ResidentRecurrentSpec)
	if rs == nil || !s.m.specRollbackSafeVia(rs, width) {
		return nil
	}
	return rs
}

// verifyWidth is the verify width a generation with opt runs at: opt.VerifyWidth, or defaultVerifyWidth, clamped to the
// drafter's block.
func (s *BlockSpec) verifyWidth(opt BlockSpecOptions) int {
	width := opt.VerifyWidth
	if width <= 0 {
		width = defaultVerifyWidth
	}
	if bs := s.dw.BlockSize(); width > bs {
		width = bs // never ask the target to verify positions the drafter did not draft
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
		return nil, 0, fmt.Errorf("decoder: block-spec verify width %d on a recurrent model needs the resident's fold path (ResidentRecurrentSpec)", width)
	}
	// Claim the single shared resident KV before any device write: this path drives the same positional cache as Model.Generate and the
	// n-gram/speculative paths. A loser returns before touching state, so the caller's fallback to plain Generate is exact. Gated on
	// m.resident != nil, mirroring model.go's useGPU check: a host without a resident keeps its KV in its own CPU cache and must not
	// contend for resBusy or forget a resIDs commit it never wrote.
	var reuseFrom int
	if m.resident != nil {
		// tryClaimResident, not a bare CAS: it also refuses while an MC3 generation holds a batch place, which never
		// sets resBusy (docs/completed/task-audit-followups-2026-10-06.md, A1).
		if !m.tryClaimResident() {
			return nil, 0, errBlockSpecResidentBusy
		}
		defer atomic.StoreInt32(&m.resBusy, 0)
		// Compute reuse before forgetting: forgetting clears the very state residentReuseLen reads. BlockSpec never binds an adapter (nil lora,
		// matching residentCommitIDs' own nil below).
		// MC1: binds the KV slot (clearing resDrafterSynced on a switch). Off: the drafter keeps its own context, which a
		// copy of another sequence's rows would not bring with it.
		sc := opt.scope
		sc.Off = true
		reuseFrom = m.residentAcquireIn(prompt, nil, nil, nil, sc).reuse
		// resIDs matching is NOT enough on its own: a plain Generate or n-gram-speculative turn
		// can commit resIDs without ever touching THIS drafter's own context, so the token
		// prefix can match while rd's state does not reflect it at all. Only trust reuseFrom
		// when the last resident write was this same *BlockSpec's own completed generate().
		if m.resDrafterSynced != s {
			reuseFrom = 0
		}
		// Forget FIRST — from here until the generation completes (or this call returns early)
		// the resident KV is mid-write, so the next turn must cold-prefill rather than trust it
		// (decoder/resident_reuse.go).
		m.residentForgetIDs()
	}
	if err := host.SetBatchedCapture(s.taps); err != nil {
		return nil, 0, err
	}
	defer func() { _ = host.SetBatchedCapture(nil) }()
	// Keep the drafter's own context for the reused prefix instead of wiping to 0: its positions correspond 1:1 with the target's (each
	// prompt token appends exactly one fused row via fuse, below), so TruncateContext(reuseFrom) is the drafter-side twin of the target's
	// resident-KV reuse. reuseFrom is 0 (a full wipe) whenever nothing is safely reusable.
	rd.TruncateContext(reuseFrom)

	hidden := m.w.arch.HiddenDim
	eos := blockSpecStopSet(m, opt)
	// fuse folds a batched capture into the drafter's context: the taps for n rows become n
	// concatenated rows, projected and appended.
	fuse := func(capt [][]float32, n int) error {
		cat := make([][]float32, n)
		for i := range n {
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

	suffix := prompt[reuseFrom:]
	embs := make([][]float32, len(suffix))
	for i, id := range suffix {
		embs[i] = m.embedResident(id)
	}
	// The seed reads one id: ask for one when the backend can (ResidentSeedArgmax), which skips an M x vocab logits buffer, copy and host argmax.
	var anchor int
	if seeder, ok := host.(ResidentSeedArgmax); ok {
		anchor, err = seeder.PrefillSeedArgmax(embs, reuseFrom)
		if err != nil {
			return nil, 0, err
		}
	} else {
		ids, e := host.PrefillLastNArgmax(embs, reuseFrom)
		if e != nil {
			return nil, 0, e
		}
		anchor = ids[len(ids)-1]
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
	for opt.MaxTokens <= 0 || len(out) < opt.MaxTokens {
		if eos[anchor] {
			break
		}
		if guard.stopped {
			// The drafter is not paying for itself on this generation: finish with plain resident decoding. Disarm the capture seam first: left
			// armed, every fallback token pays tap downloads for hidden states nothing will read, and the guard then costs more than it saves.
			if !seamOff {
				if e := host.SetBatchedCapture(nil); e != nil {
					return out, rounds, e
				}
				seamOff = true
			}
			// Use the fast greedy step where the backend has one (argmax reduced on-device with a 4-byte readback, as Model.Generate does).
			// Falling back through PrefillLastNArgmax(M=1) downloads the full logit row per token, so the guard would turn a loss into the same
			// loss instead of into plain-decode speed.
			var e error
			if g, ok := m.resident.(ResidentGreedy); ok {
				anchor, e = g.ForwardArgmax(m.embedResident(anchor), pos)
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
		if fold != nil {
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
	// The only exit that commits, mirroring generateInto's rule that a generation which ran to completion is the only place the resident
	// cache's contents are recorded (model.go). Every early return above (a device error, or the caller's emit stopping consumption) leaves
	// resIDs nil, so the next turn cold-prefills rather than trusting a state this function cannot vouch for. resDrafterSynced = s marks this
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
				} else if g, ok := m.resident.(ResidentGreedy); ok {
					_, e = g.ForwardArgmax(emb, pos)
				} else {
					_, e = host.PrefillLastNArgmax([][]float32{emb}, pos)
				}
				if e == nil {
					keep = len(out)
				}
			}
		}
		m.residentCommitIDs(prompt, out[:keep], nil, nil) // BlockSpec never binds an adapter
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
	if !s.m.specRollbackSafeVia(s.recurrentFold(s.verifyWidth(BlockSpecOptions{})), s.verifyWidth(BlockSpecOptions{})) {
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
