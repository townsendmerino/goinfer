package decoder

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/specdecode"
)

var (
	errBlockSpecUnsupported   = specdecode.ErrUnsupported
	errBlockSpecResidentBusy  = specdecode.ErrResidentBusy
	blockSpecGuardTripForTest bool
)

// ErrBlockSpecUnsupported reports whether err is the unsupported-backend decline.
func ErrBlockSpecUnsupported(err error) bool { return errors.Is(err, specdecode.ErrUnsupported) }

// ErrBlockSpecResidentBusy reports whether err is the concurrent-claim decline.
func ErrBlockSpecResidentBusy(err error) bool { return errors.Is(err, specdecode.ErrResidentBusy) }

// RecurrentSpecMaxRows is the widest verify a RecurrentSpecAvailable arena is sized for.
const RecurrentSpecMaxRows = specdecode.RecurrentSpecMaxRows

const (
	defaultVerifyWidth = specdecode.DefaultVerifyWidth
)

// RecurrentSpecArenaBytes is the device memory a RecurrentSpecAvailable resident keeps for verify passes of up to rows rows.
func RecurrentSpecArenaBytes(a *darch.Architecture, rows int) int64 {
	return specdecode.RecurrentArenaBytes(a, rows)
}

func recurrentSpecReserve(a *darch.Architecture, drafter bool) int64 {
	if !drafter {
		return 0
	}
	return specdecode.RecurrentArenaBytes(a, RecurrentSpecMaxRows)
}

// BlockSpecCapable reports whether this model can run block-drafting speculation.
func (m *Model) BlockSpecCapable() bool {
	return m.resident != nil && m.resRuntime().DrafterHostAvailable()
}

// BlockSpecOptions tunes block-drafting speculation.
type BlockSpecOptions struct {
	VerifyWidth int
	MaxTokens   int
	StopIDs     []int
	OnRound     func(width, committed int)
	scope       KVScope
}

// BlockSpec is an attached block drafter, ready to serve many generations.
type BlockSpec struct {
	m     *Model
	host  ResidentDrafterHost
	rd    ResidentBlockDrafter
	dw    BlockDrafterWeights
	taps  []int
	width int
}

// VerifyWidthInUse is the verify width GenerateStream and default Generate run at.
func (s *BlockSpec) VerifyWidthInUse() int {
	return s.verifyWidth(BlockSpecOptions{})
}

// SetVerifyWidth sets the verify width GenerateStream and default Generate run at.
func (s *BlockSpec) SetVerifyWidth(w int) error {
	if w != 0 && w < 2 {
		return fmt.Errorf("decoder: block-spec verify width %d: want 0 (the default) or at least 2", w)
	}
	v := w
	if v == 0 {
		v = defaultVerifyWidth
	}
	v = min(v, s.dw.BlockSize())
	if !s.m.specRollbackSafeVia(s.recurrentFold(v), v) {
		return fmt.Errorf("decoder: block-spec verify width %d (%d after the drafter's block of %d) cannot roll back on this model (a recurrent model past its resident's fold rows, or a staged sliding window)",
			w, v, s.dw.BlockSize())
	}
	s.width = w
	return nil
}

// NewBlockSpec attaches a block drafter to this model's resident target.
func (m *Model) NewBlockSpec(dw BlockDrafterWeights, taps []int) (*BlockSpec, error) {
	_, done, err := m.enterCall(context.Background())
	if err != nil {
		return nil, err
	}
	defer done()
	out, err := m.newBlockSpec(dw, taps)
	return out, m.closeErr(context.Background(), err)
}

func (m *Model) newBlockSpec(dw BlockDrafterWeights, taps []int) (*BlockSpec, error) {
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
	if err := m.prepareRecurrentSpec(host, dw); err != nil {
		return nil, err
	}
	rd, err := rt.AttachBlockDrafter(dw)
	if err != nil {
		return nil, err
	}
	return &BlockSpec{m: m, host: host, rd: rd, dw: dw, taps: taps}, nil
}

func (m *Model) blockSpecHost() (host ResidentDrafterHost, rt ResidentRuntimeSubsystem, ok bool) {
	rt = m.resRuntime()
	if !rt.DrafterHostAvailable() {
		return nil, nil, false
	}
	host, ok = drafterHostOf(m.resident)
	return host, rt, ok
}

func (m *Model) prepareRecurrentSpec(host ResidentDrafterHost, dw BlockDrafterWeights) error {
	if !m.hasRecurrentState() {
		return nil
	}
	rt := SubsystemsOf(host).Runtime
	if !rt.RecurrentSpecAvailable() {
		return fmt.Errorf("decoder: this model has recurrent state and its resident cannot verify without writing it (no RecurrentSpecAvailable)")
	}
	if RecurrentSpecArenaBytes(m.w.Arch, 1) == 0 {
		return fmt.Errorf("decoder: the fold path does not serve this model (a recurrent layer that is not Gated DeltaNet, heads past 128, or a sliding window)")
	}
	rows := min(dw.BlockSize(), RecurrentSpecMaxRows)
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

func (s *BlockSpec) verifyWidth(opt BlockSpecOptions) int {
	width := opt.VerifyWidth
	if width <= 0 {
		width = s.width
	}
	if width <= 0 {
		width = defaultVerifyWidth
	}
	if bs := s.dw.BlockSize(); width > bs {
		width = bs
	}
	if s.m.hasRecurrentState() && width > RecurrentSpecMaxRows {
		width = RecurrentSpecMaxRows
	}
	return width
}

// Generate runs greedy generation with the attached drafter.
func (s *BlockSpec) Generate(prompt []int, opt BlockSpecOptions) ([]int, int, error) {
	return s.generate(prompt, opt, nil)
}

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

func (s *BlockSpec) generate(prompt []int, opt BlockSpecOptions, emit func([]int) bool) (out []int, rounds int, err error) {
	m, host, rd, dw := s.m, s.host, s.rd, s.dw
	defer m.enterExactAttention()()
	width := s.verifyWidth(opt)
	if width < 2 {
		return nil, 0, fmt.Errorf("decoder: block-spec verify width %d is too narrow", width)
	}
	if ctxCap := m.ResidentContextCap(); ctxCap > 0 {
		switch left := ctxCap - len(prompt); {
		case left == 0:
			return nil, 0, nil
		case left > 0 && (opt.MaxTokens <= 0 || opt.MaxTokens > left):
			opt.MaxTokens = left
		}
	}
	fold := s.recurrentFold(width)
	if m.hasRecurrentState() && fold == nil {
		return nil, 0, fmt.Errorf("decoder: block-spec verify width %d on a recurrent model needs the resident's fold path (RecurrentSpecAvailable)", width)
	}
	if fold != nil {
		defer fold.DropFold()
	}
	var reuseFrom int
	if m.resident != nil {
		if !m.tryClaimResident() {
			return nil, 0, errBlockSpecResidentBusy
		}
		defer atomic.StoreInt32(&m.resBusy, 0)
		sc := opt.scope
		sc.Off = true
		reuseFrom = m.resHost().AcquireIn(prompt, nil, 0, nil, sc).Reuse
		if m.resDrafterSynced != s {
			reuseFrom = 0
		}
		m.resHost().ForgetIDs()
	}
	if err := host.SetBatchedCapture(s.taps); err != nil {
		return nil, 0, err
	}
	defer func() { _ = host.SetBatchedCapture(nil) }()
	rd.TruncateContext(reuseFrom)

	var fallbackRow [1][]float32
	env := specdecode.BlockSpecEnv{
		Host:          host,
		RD:            rd,
		Fold:          fold,
		HiddenDim:     m.w.Arch.HiddenDim,
		MaskID:        dw.MaskTokenID(),
		Width:         width,
		MaxTokens:     opt.MaxTokens,
		ReuseFrom:     reuseFrom,
		EOS:           blockSpecStopSet(m, opt),
		GuardTrip:     blockSpecGuardTripForTest,
		Embed:         m.embedResident,
		SeedAnchor:    s.seedAnchor,
		CanFastGreedy: m.canDecode(outArgmax),
		FastGreedy: func(emb []float32, pos int) (int, error) {
			fallbackRow[0] = emb
			sr, e := m.step(stepBatch{Pass: passToken, Emb: fallbackRow[:], Pos: pos, Out: outArgmax})
			return sr.ID, e
		},
		CtxCap:  m.ResidentContextCap,
		OnRound: opt.OnRound,
		Emit:    emit,
	}
	out, rounds, pos, seamOff, completed, err := specdecode.RunBlockSpec(env, prompt)
	if err != nil || !completed {
		return out, rounds, err
	}
	if fold != nil {
		if err := fold.FlushFold(); err != nil {
			return out, rounds, err
		}
	}
	if m.resident != nil {
		keep := specdecode.FinishBlockSpecTrailing(env, prompt, out, pos, seamOff)
		m.resHost().CommitIDs(prompt, out[:keep], nil, 0)
		m.resDrafterSynced = s
	}
	return out, rounds, nil
}

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

// GenerateStream is the serving-shaped entry point for greedy block-drafting speculation.
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
	leaveExact := s.m.enterExactAttention()
	s.m.runGen(ctx, out, g, leaveExact, func(ctx context.Context) {
		defer leaveExact()
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
		var drafted, evaluated int
		opt := BlockSpecOptions{
			MaxTokens: maxTokens,
			StopIDs:   sp.StopIDs,
			scope:     kvScopeOf(ctx),
			OnRound: func(width, committed int) {
				drafted += width - 1
				evaluated += committed
			},
		}
		toks, rounds, err := s.generate(prompt, opt, emit)
		stats.Rounds = rounds
		stats.Emitted = len(toks)
		stats.Drafted = drafted
		stats.Evaluated = evaluated
		if a := len(toks) - rounds - 1; a > 0 {
			stats.Accepted = a
		}
		if err != nil {
			g.err = err
		} else if ctx.Err() != nil {
			g.err = ctx.Err()
		}
	})
	return out, g, nil
}
