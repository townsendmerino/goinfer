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

// ResidentSeedArgmax is an OPTIONAL narrowing of PrefillLastNArgmax for the prompt seed: the one
// place the loop asks for M rows of argmax and reads exactly one of them.
//
// The seed calls PrefillLastNArgmax over the whole prompt and uses `ids[len(ids)-1]`. On a
// vocab-151,936 target a 2048-token prompt therefore allocates 1.24 GB of VRAM for the batched
// logits, a 1.24 GB host slice and a 1.24 GB device-to-host copy, runs the head GEMV over 2048 rows
// and a single-threaded host argmax over 311M floats — to obtain ONE token id. The capture the seed
// actually needs comes from the LAYER LOOP, not the head, so heading one row loses nothing
// (audit-2026-09-02 C-12).
//
// Optional rather than added to ResidentDrafterHost, following ResidentCapped/ResidentGreedy: a
// backend that has not implemented it keeps working through the wide path.
type ResidentSeedArgmax interface {
	// PrefillSeedArgmax runs the same batched forward and returns only the LAST row's argmax.
	// The batched capture seam must be armed and filled exactly as PrefillLastNArgmax fills it.
	PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error)
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

	// StopIDs are the caller's extra stop tokens — SamplingParams.StopIDs, which for a served
	// request carries the CHAT TEMPLATE's stops on top of the model's own.
	//
	// It exists because this loop rebuilt its stop set from Cfg.EOSIDs() alone while every other
	// speculative loop asks target.isStop(tok, sp). For the pairing this ships for (Qwen3-4B +
	// DFlash) that is {151645} against m.eosIDs' {151645, 151643}, so a <|endoftext|> was emitted
	// as ordinary content and generation ran on to <|im_end|> or max_tokens — with streamTokens
	// decoding the stop token into the response. The "lossless by construction" contract was
	// broken by the STOP SET, not by the verify (audit-2026-09-02 C-11).
	StopIDs []int

	// OnRound, if non-nil, is called after each completed round with the verify width that
	// round used and the tokens it committed (accepted drafts plus the target's own token).
	//
	// It is one seam serving two needs the campaign has: the adaptive controller cannot be
	// judged on end-to-end speed alone — a regime change mid-generation is exactly where a
	// cumulative-average signal lags, and an aggregate number hides that — and `serve` owes
	// per-request accept-rate and tok/verify in its response metadata regardless.
	OnRound func(width, committed int)
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

	embs := make([][]float32, len(prompt))
	for i, id := range prompt {
		embs[i] = m.embedResident(id)
	}
	// C-12: the seed reads ONE id. Ask for one when the backend can, which skips an M x vocab
	// device buffer, host slice, D2H copy and host argmax — gigabytes at prompt lengths over ~1k.
	var anchor int
	if seeder, ok := host.(ResidentSeedArgmax); ok {
		anchor, err = seeder.PrefillSeedArgmax(embs, 0)
		if err != nil {
			return nil, 0, err
		}
	} else {
		ids, e := host.PrefillLastNArgmax(embs, 0)
		if e != nil {
			return nil, 0, e
		}
		anchor = ids[len(ids)-1]
	}
	if err := fuse(host.BatchedCapture(), len(prompt)); err != nil {
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
			// The FAST greedy step where the backend has one: argmax reduced on-device with a
			// 4-byte readback, which is what Model.Generate uses. Falling back through
			// PrefillLastNArgmax(M=1) instead downloads the full logit row per token — the
			// same slow primitive that made gate 3's baseline wrong, and it left the guard
			// converting a 0.82x into 0.82x instead of into plain-decode speed.
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
			// The stop token is NOT emitted here either. The loop-top check breaks on it, but
			// only AFTER this append ran — so without this the fallback emits one token past
			// where plain decoding stops, which is exactly the 259-vs-258 mismatch.
			if eos[anchor] {
				break
			}
			out = append(out, anchor)
			if emit != nil && !emit([]int{anchor}) {
				return out, rounds, nil
			}
			continue
		}
		// M-13: the round's width is clamped by what is left of BOTH budgets.
		//
		// MaxTokens: the loop condition is checked per ROUND while a round commits up to `width`
		// tokens at once, so max_tokens=2 could return 9 and usage.completion_tokens could exceed
		// the request's own cap. Neither losslessness gate could see it — the CPU one compares only
		// the common prefix, and the CUDA one asks the reference for exactly len(got) tokens.
		//
		// The context cap: verifying `width` rows at `pos` with no clamp makes checkCap refuse the
		// WHOLE round near the end of the window, so a nearly complete response ends in a
		// generation error. Plain Generate and the server both clamp instead, so a max-length turn
		// finishes cleanly with "length"; this path did not.
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
		// TRUNCATE BEFORE EOS INSIDE THE BURST. A round commits several tokens at once, so a
		// stop token can land in the MIDDLE of one; appending the whole burst emits content
		// AFTER it, which plain decoding never does. And the stop token itself is EXCLUDED,
		// because Generate breaks on it without emitting (model.go, isStop) — matching that
		// exactly is what makes the two paths token-identical.
		//
		// Caught by a 384-token run where spec emitted 259 tokens against greedy's 258. At 96
		// tokens neither generation reached EOS, so the bug was invisible — a reminder that a
		// losslessness gate only covers the lengths it actually runs.
		stop := false
		for i, id := range burst {
			if eos[id] {
				burst, stop = burst[:i], true
				break
			}
		}
		out = append(out, burst...)
		if emit != nil && !stop && !emit(burst) {
			return out, rounds, nil
		}
		pos += 1 + accepted
		if e := fuse(capt, 1+accepted); e != nil {
			return out, rounds, e
		}
		if stop {
			if emit != nil && len(burst) > 0 {
				emit(burst)
			}
			break
		}
		anchor = next
		// M-13: `width` here is the CLAMPED width this round actually verified, which is what the
		// telemetry must count — drafted = the positions offered (width-1, the anchor is not a
		// draft), evaluated = the positions tested. GenerateStream left Drafted and Evaluated at
		// zero, so AcceptanceRate() was 0 for every block-spec generation and the adaptive
		// controller's own signal was unreadable.
		if opt.OnRound != nil {
			opt.OnRound(width, 1+accepted)
		}
		guard.observe(1 + accepted)
	}
	return out, rounds, nil
}

// blockSpecStopSet is this loop's stop predicate, and it must agree with Model.isStop — the
// predicate plain decoding and every OTHER speculative loop use.
//
// It did not. The set was rebuilt from Cfg.EOSIDs() alone: config.json's eos_token_id, without
// generation_config.json's additions (which resolveEOSIDs merges into m.eosIDs) and without the
// caller's SamplingParams.StopIDs (which for a served request carries the chat template's stops).
// On the pairing this ships for, Qwen3-4B + DFlash, that is {151645} against {151645, 151643}: a
// <|endoftext|> was emitted as ordinary content and generation ran on to <|im_end|> or max_tokens,
// with streamTokens decoding the stop token into the response. "Lossless by construction" was
// broken by the STOP SET, not by the verify (audit-2026-09-02 C-11).
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

// blockSpecRoundWidth clamps a round's verify width to what is left of BOTH budgets. Returns < 1
// when neither has room, which the caller treats as a clean finish.
//
// TWO SEPARATE M-13 DEFECTS, ONE CLAMP. The loop tests its token budget once per ROUND while a
// round commits up to `width` tokens at once, so max_tokens=2 could return 9 and
// usage.completion_tokens could exceed the request's own cap. And verifying `width` rows at `pos`
// with no context clamp made the backend's checkCap refuse the WHOLE round near the end of the
// window, so a nearly complete response ended in a generation error — where plain Generate and the
// server both clamp instead, and a max-length turn finishes cleanly with "length".
//
// Neither losslessness gate could see either one: the CPU gate compares only the common prefix, and
// the CUDA gate asks the reference for exactly len(got) tokens.
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
		// StopIDs: this loop is the one speculative path that did not consult the caller's stop
		// set, so a chat-template stop was decoded into the response as content (C-11).
		//
		// Drafted/Evaluated come from the per-round hook because only the loop knows the CLAMPED
		// width each round used; computing them out here from len(toks) would assume every round
		// ran at the configured width, which is exactly what the M-13 clamp makes untrue.
		var drafted, evaluated int
		opt := BlockSpecOptions{
			MaxTokens: maxTokens,
			StopIDs:   sp.StopIDs,
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
			// M-13: a cancelled generation returned with g.err nil, so the caller read a truncated
			// stream as a clean finish. Same shape as M-23 one tranche earlier, in a different loop.
			g.err = ctx.Err()
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
// 2.5, BELOW break-even, not above it. This was 3.8 and that was backwards. The reasoning for a
// margin ABOVE break-even was "disabling a drafter that is merely breaking even costs nothing" —
// which is false, because acceptance MEASURED OVER THE FIRST FEW ROUNDS is not acceptance over
// the generation. Math averages 5.88 tok/round end to end and is a 1.58x workload, but its
// opening rounds are slow enough that a 3.8 threshold disabled it: 1.58x became 0.97x, the guard
// costing 39% on a workload it was supposed to protect.
//
// So the margin belongs BELOW break-even. A false negative (disabling a paying workload) costs
// ~40%; a false positive (six unprofitable rounds before tripping) costs ~8%. The guard should
// only fire when a workload is CLEARLY losing — chat sits at 1.96 and still trips at 2.5 — and
// should leave anything ambiguous alone.
const breakEvenTokensPerRound = 2.5

// guardWindow is how many rounds to observe before judging.
//
// SIX. Three was TRIED AND REVERTED, and the measurement is worth keeping because it refutes
// the reasoning that motivated it:
//
//	case      window 6   window 3
//	code        1.57x      1.54x   (kept either way)
//	MATH        1.58x      0.91x   <- falsely tripped
//	chat A      0.91x      0.90x
//	chat B      0.94x      0.96x
//	thinking    0.96x      0.92x
//
// Three bought nothing on chat and cost 42% on math by disabling a drafter that was paying.
// The argument for shortening was that a response OPENS with boilerplate, so early rounds are
// optimistic and judging early errs toward keeping a good drafter. That is false for math,
// whose opening is evidently less predictable than its body — the whole-generation average
// (5.88 tok/round) hides a slow start.
//
// It also confirms the asymmetry that shapes this whole design: a false negative costs ~42%
// (a paying workload disabled) where a false positive costs ~8% (six unprofitable rounds).
// The window should err LONG. Six, not twelve, because the rounds spent deciding are pure loss when the answer is "stop":
// twelve rounds is a third of a 96-token response, and halving the window halves that. The risk
// of judging early is disabling a drafter that would have paid — and that risk is LOW here in a
// way worth stating: a response's opening is boilerplate ("Here's a Python function...", a code
// fence), which is the part a drafter predicts BEST. Early rounds are optimistic, so a short
// window errs toward keeping a good drafter, not dropping one.
const guardWindow = 6

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
	// CUMULATIVE, not per-window. Resetting the counters after a passing window gave a losing
	// workload a fresh budget every time: chat survived its first six rounds, reset, and only
	// tripped on the twelfth — measured 0.79x where tripping at six gives 0.91x. Keeping the
	// running average means a workload trips as soon as its EVIDENCE says so, while a
	// slow-starting profitable one (math opens below its own average) recovers as later rounds
	// pull the average up, instead of being judged on a six-round snapshot.
	if g.rounds >= guardWindow {
		if float64(g.tokens)/float64(g.rounds) < breakEvenTokensPerRound {
			g.stopped = true
			return false
		}
	}
	return true
}
