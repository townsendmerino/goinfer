package specdecode

import (
	"errors"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// ErrUnsupported is returned when the backend cannot host a block drafter.
var ErrUnsupported = errors.New("decoder: backend does not support block-drafting speculation")

// ErrResidentBusy is returned when another generation already holds the model's single shared resident KV.
var ErrResidentBusy = errors.New("decoder: resident KV is busy with another generation")

// RecurrentSpecMaxRows is the widest verify a RecurrentSpecAvailable arena is sized for.
const RecurrentSpecMaxRows = 16

// BlockSpecFuseRows is the most rows one FuseContext call takes.
const BlockSpecFuseRows = 512

// DefaultVerifyWidth is the default verify width for block-drafting speculation.
const DefaultVerifyWidth = 8

// BreakEvenTokensPerRound is the acceptance below which the guard disables block drafting.
const BreakEvenTokensPerRound = 2.5

// GuardWindow is how many rounds to observe before judging block-drafting acceptance.
const GuardWindow = 6

// RecurrentArenaBytes is the device memory a RecurrentSpecAvailable resident keeps for verify passes of up to rows rows.
func RecurrentArenaBytes(a *darch.Architecture, rows int) int64 {
	if a == nil || a.Qwen35 == nil || rows < 1 || a.SlidingWindow > 0 || a.Qwen35.KeyHeadDim > 128 || a.Qwen35.ValueHeadDim > 128 {
		return 0
	}
	nDN := 0
	for l := range a.NumLayers {
		for _, o := range a.LayerOps(l, false) {
			switch {
			case o.Kind == program.OpDeltaNet:
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

// BlockSpecRoundWidth clamps a round's verify width to what is left of both budgets.
func BlockSpecRoundWidth(width, maxTokens, emitted, pos, ctxCap int) int {
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

// AcceptanceGuard disables a block drafter that is not paying for itself, per generation.
type AcceptanceGuard struct {
	rounds  int
	tokens  int
	stopped bool
	trip    bool
}

// NewAcceptanceGuard constructs an AcceptanceGuard (tripForTest forces tripping at GuardWindow).
func NewAcceptanceGuard(tripForTest bool) AcceptanceGuard {
	return AcceptanceGuard{trip: tripForTest}
}

// Stopped reports whether the guard has tripped and fallen back to plain decode.
func (g *AcceptanceGuard) Stopped() bool { return g.stopped }

// Observe records a round and reports whether block drafting should continue.
func (g *AcceptanceGuard) Observe(committed int) bool {
	if g.stopped {
		return false
	}
	g.rounds++
	g.tokens += committed
	if g.rounds >= GuardWindow {
		if g.trip || float64(g.tokens)/float64(g.rounds) < BreakEvenTokensPerRound {
			g.stopped = true
			return false
		}
	}
	return true
}

// BlockSpecEnv carries the narrow host operations RunBlockSpec needs to drive a block-drafting generation loop.
type BlockSpecEnv struct {
	Host          resident.DrafterHost
	RD            resident.BlockDrafter
	Fold          resident.RuntimeSubsystem // nil when not recurrent
	HiddenDim     int
	MaskID        int
	Width         int
	MaxTokens     int
	ReuseFrom     int
	EOS           map[int]bool
	GuardTrip     bool
	Embed         func(id int) []float32
	SeedAnchor    func(embs [][]float32, pos int) (int, error)
	CanFastGreedy bool
	FastGreedy    func(emb []float32, pos int) (int, error)
	CtxCap        func() int
	OnRound       func(width, committed int)
	Emit          func([]int) bool
}

func fuseCaptures(rd resident.BlockDrafter, hidden int, capt [][]float32, n int) error {
	for lo := 0; lo < n; lo += BlockSpecFuseRows {
		hi := min(lo+BlockSpecFuseRows, n)
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

// RunBlockSpec executes the core block-drafting speculative generation loop (after capture setup and resident claim).
// completed is true only when the generation ran to completion (not stopped early by emit returning false or immediate seed EOS).
func RunBlockSpec(env BlockSpecEnv, prompt []int) (out []int, rounds int, pos int, seamOff bool, completed bool, err error) {
	host, rd, fold := env.Host, env.RD, env.Fold
	suffix := prompt[env.ReuseFrom:]
	embs := make([][]float32, len(suffix))
	for i, id := range suffix {
		embs[i] = env.Embed(id)
	}
	anchor, err := env.SeedAnchor(embs, env.ReuseFrom)
	if err != nil {
		return nil, 0, 0, false, false, err
	}
	if err := fuseCaptures(rd, env.HiddenDim, host.BatchedCapture(), len(suffix)); err != nil {
		return nil, 0, 0, false, false, err
	}
	pos = len(prompt)
	if env.EOS[anchor] {
		return out, rounds, pos, false, false, nil
	}
	out = append(out, anchor)
	if env.Emit != nil && !env.Emit([]int{anchor}) {
		return out, rounds, pos, false, false, nil
	}

	guard := NewAcceptanceGuard(env.GuardTrip)
	for env.MaxTokens <= 0 || len(out) < env.MaxTokens {
		if env.EOS[anchor] {
			break
		}
		if guard.Stopped() {
			if !seamOff {
				if fold != nil {
					if e := fold.FlushFold(); e != nil {
						return out, rounds, pos, seamOff, false, e
					}
				}
				if e := host.SetBatchedCapture(nil); e != nil {
					return out, rounds, pos, seamOff, false, e
				}
				seamOff = true
			}
			var e error
			if env.CanFastGreedy {
				anchor, e = env.FastGreedy(env.Embed(anchor), pos)
			} else {
				var one []int
				one, e = host.PrefillLastNArgmax([][]float32{env.Embed(anchor)}, pos)
				if e == nil {
					anchor = one[0]
				}
			}
			if e != nil {
				return out, rounds, pos, seamOff, false, e
			}
			pos++
			if env.EOS[anchor] {
				break
			}
			out = append(out, anchor)
			if env.Emit != nil && !env.Emit([]int{anchor}) {
				return out, rounds, pos, seamOff, false, nil
			}
			continue
		}
		w := BlockSpecRoundWidth(env.Width, env.MaxTokens, len(out), pos, env.CtxCap())
		if w < 1 {
			break
		}
		blockIn := make([][]float32, w)
		blockIn[0] = env.Embed(anchor)
		for i := 1; i < w; i++ {
			blockIn[i] = env.Embed(env.MaskID)
		}
		trunk, e := rd.DraftBlock(blockIn)
		if e != nil {
			return out, rounds, pos, seamOff, false, e
		}
		drafted, e := rd.DraftTokens(trunk[1:])
		if e != nil {
			return out, rounds, pos, seamOff, false, e
		}
		vin := make([][]float32, 0, 1+len(drafted))
		vin = append(vin, env.Embed(anchor))
		for _, id := range drafted {
			vin = append(vin, env.Embed(id))
		}
		var tgt []int
		if fold != nil {
			tgt, e = fold.VerifyLastNArgmax(vin, pos)
		} else {
			tgt, e = host.PrefillLastNArgmax(vin, pos)
		}
		if e != nil {
			return out, rounds, pos, seamOff, false, e
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
		next := tgt[accepted]
		burst = append(burst, next)
		stop := false
		for i, id := range burst {
			if env.EOS[id] {
				burst, stop = burst[:i], true
				break
			}
		}
		rows := 1 + min(accepted, len(burst))
		if fold != nil {
			if e := fold.FoldVerified(rows); e != nil {
				return out, rounds, pos, seamOff, false, e
			}
		}
		out = append(out, burst...)
		if env.Emit != nil && !stop && !env.Emit(burst) {
			return out, rounds, pos, seamOff, false, nil
		}
		pos += rows
		if e := fuseCaptures(rd, env.HiddenDim, capt, rows); e != nil {
			return out, rounds, pos, seamOff, false, e
		}
		if stop {
			if env.Emit != nil && len(burst) > 0 {
				env.Emit(burst)
			}
			break
		}
		anchor = next
		if env.OnRound != nil {
			env.OnRound(w, 1+accepted)
		}
		guard.Observe(1 + accepted)
	}
	return out, rounds, pos, seamOff, true, nil
}

// FinishBlockSpecTrailing forwards the single trailing emitted token if room permits, returning how many tokens of out are committed.
func FinishBlockSpecTrailing(env BlockSpecEnv, prompt, out []int, pos int, seamOff bool) int {
	keep := len(out)
	if len(prompt)+len(out) <= pos {
		return keep
	}
	keep = max(0, pos-len(prompt))
	ctxCap := env.CtxCap()
	if len(prompt)+len(out) == pos+1 && (ctxCap <= 0 || pos < ctxCap) {
		emb := env.Embed(out[len(out)-1])
		var e error
		if !seamOff {
			if _, e = env.Host.PrefillLastNArgmax([][]float32{emb}, pos); e == nil {
				e = fuseCaptures(env.RD, env.HiddenDim, env.Host.BatchedCapture(), 1)
			}
		} else if env.CanFastGreedy {
			_, e = env.FastGreedy(emb, pos)
		} else {
			_, e = env.Host.PrefillLastNArgmax([][]float32{emb}, pos)
		}
		if e == nil {
			keep = len(out)
		}
	}
	return keep
}
