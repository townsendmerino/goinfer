package decoder

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
)

var (
	cpuHooksMu     sync.Mutex
	decodeTimingSt = cpuengine.DecodeTimingStats{
		Attn: &dtAttn, MLP: &dtMLP, Head: &dtHead, GU: &dtGU,
		Act: &dtAct, ActDown: &dtActDown, QKV: &dtQKV, O: &dtO, AttnAll: &dtAttnAll,
	}
)

func init() { syncCPUHooks() }

// syncCPUHooks publishes decoder's diagnostic and test hook globals to cpuengine only when changed
// (zero atomic writes on the steady-state hot path, safe under -race).
func syncCPUHooks() {
	cur := cpuengine.ActiveHooks()
	ssmOn := ssmStopLayer >= 0
	if cur != nil &&
		cur.KVRoundF16 == kvRoundF16ForTest &&
		cur.KVF16NewRowsF32 == kvF16NewRowsF32ForTest &&
		cur.KVF16RingOffset == kvF16RingOffsetForTest &&
		cur.ConvWinSlice == convWinSliceForTest &&
		cur.Int4FloatAct == int4FloatActForTest &&
		cur.CPUFusedGateUp == cpuFusedGateUp &&
		cur.DeltaNetTiming == deltaNetTiming &&
		cur.DecodeTiming == decodeTiming &&
		cur.W4A8BatchEnabled == w4a8BatchEnabled &&
		cur.W4A8PreOff == w4a8PreOff &&
		cur.W4A8PreCalls == &w4a8PreCalls &&
		cur.AttnScoreStats == &attnScoreStatsForTest &&
		cur.SeamPlant == seamPlantForTest &&
		cur.WalkBatchPlant == walkBatchPlantForTest &&
		cur.SSMStopEnabled == ssmOn &&
		cur.SSMStopLayer == ssmStopLayer &&
		(cur.G4TraceHidden != nil) == (g4traceHidden != nil) &&
		(cur.G4TraceQKV != nil) == (g4traceQKV != nil) &&
		cur.ReleaseHostMemory != nil &&
		cur.DecodeTimingStats == &decodeTimingSt {
		return
	}
	cpuHooksMu.Lock()
	defer cpuHooksMu.Unlock()
	h := &cpuengine.Hooks{
		KVRoundF16:        kvRoundF16ForTest,
		KVF16NewRowsF32:   kvF16NewRowsF32ForTest,
		KVF16RingOffset:   kvF16RingOffsetForTest,
		ConvWinSlice:      convWinSliceForTest,
		Int4FloatAct:      int4FloatActForTest,
		CPUFusedGateUp:    cpuFusedGateUp,
		DeltaNetTiming:    deltaNetTiming,
		DecodeTiming:      decodeTiming,
		W4A8BatchEnabled:  w4a8BatchEnabled,
		W4A8PreOff:        w4a8PreOff,
		W4A8PreCalls:      &w4a8PreCalls,
		AttnScoreStats:    &attnScoreStatsForTest,
		SeamPlant:         seamPlantForTest,
		WalkBatchPlant:    walkBatchPlantForTest,
		SSMStopEnabled:    ssmOn,
		SSMStopLayer:      ssmStopLayer,
		G4TraceHidden:     g4traceHidden,
		G4TraceQKV:        g4traceQKV,
		ReleaseHostMemory: releaseHostMemory,
		DecodeTimingStats: &decodeTimingSt,
		DefaultKnobs:      (*knobSet)(nil),
	}
	cpuengine.SyncHooks(h)
}

func (m *Model) cpuEngine() cpuengine.Engine {
	syncCPUHooks()
	var pager cpuengine.ExpertPager
	if m.pager != nil {
		pager = m.pager
	}
	var lp cpuengine.LayerPager
	if m.layerPager != nil {
		lp = m.layerPager
	}
	var ks cpuengine.KnobSource
	if m.knobs != nil {
		ks = m.knobs
	}
	return cpuengine.Engine{
		W:                 m.w,
		Prog:              m.prog,
		Be:                m.be,
		Knobs:             ks,
		ActQuantGroup:     m.opts.ActQuantGroup,
		ExactPrefill:      m.opts.ExactPrefill,
		Pager:             pager,
		LayerPager:        lp,
		BuildProgramCalls: &buildProgramCalls,
	}
}

func (m *Model) program() *layerProgram {
	e := m.cpuEngine()
	return e.Program()
}

func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayers(id, unwrapCache(cache))
}

func (m *Model) runLayersFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayersFromEmbed(h, unwrapCache(cache))
}

func (m *Model) runLayersFromEmbedPLE(h []float32, pleID int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.RunLayersFromEmbedPLE(h, pleID, unwrapCache(cache))
}

func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.Forward(id, unwrapCache(cache))
}

func (m *Model) forwardFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	e := m.cpuEngine()
	return e.ForwardFromEmbed(h, unwrapCache(cache))
}

func (m *Model) logitsFromHidden(h []float32, cache *KVCache) []float32 {
	e := m.cpuEngine()
	return e.LogitsFromHidden(h, unwrapCache(cache))
}

func (m *Model) logitsFromNormed(h []float32, cache *KVCache) []float32 {
	e := m.cpuEngine()
	return e.LogitsFromNormed(h, unwrapCache(cache))
}

func (m *Model) finalNorm(h []float32) {
	e := m.cpuEngine()
	e.FinalNorm(h)
}

func (m *Model) countHead(cache *KVCache, n int, t0 time.Time) {
	e := m.cpuEngine()
	e.CountHead(unwrapCache(cache), n, t0)
}

func (m *Model) gemma4PLEInputs(h []float32, id int, out []float32) {
	e := m.cpuEngine()
	e.Gemma4PLEInputs(h, id, out)
}

func (m *Model) gemma4PLEInputsScratch(h []float32, id int, out, tmp []float32) {
	e := m.cpuEngine()
	e.Gemma4PLEInputsScratch(h, id, out, tmp)
}

func (m *Model) forwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	e := m.cpuEngine()
	return e.ForwardCapture(id, unwrapCache(cache), layers)
}

// ForwardCapture runs one forward for token id and returns the next-token logits plus the residual stream after each layer in
// `layers` (cloned): the read-only hidden-state seam a draft head reads (block drafters). The forward is byte-identical to
// forward(id); the captures are copies that never feed back. Layer indices are 0-based into [0, NumLayers); out[i]
// corresponds to layers[i].
//
// Every family's layers run through the program walker, which captures the residual after each layer (gate C of §4.8's
// seams PR holds it to the forward cut after that layer, bit for bit).
func (m *Model) ForwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	_, done, err := m.enterCall(context.Background())
	if err != nil {
		return nil, nil, err
	}
	defer done()
	logits, hidden, err = m.forwardCapture(id, cache, layers)
	return logits, hidden, m.closeErr(context.Background(), err)
}

// ForwardSubCapture runs one token and returns, per layer, the attention contribution and the
// MLP contribution to the residual (scr.sub after each sublayer's sandwich-norm, before the add).
// This is the finer seam ForwardCapture doesn't give: it separates the two sublayers, which is
// what localizing a channel's sign flip to attention-vs-MLP at a specific layer requires.
// Diagnostic — same byte-identical-output contract as ForwardCapture. A single-operation block (Nemotron-H) fills one of the
// two; gate S of §4.8's seams PR rebuilds each layer's residual from them. Refused where a layer has an operation they do not
// cover (program.subCaptureUncovered).
func (m *Model) ForwardSubCapture(id int, cache *KVCache) (attn, mlp, ctx, mlpPre [][]float32, err error) {
	a := m.w.Arch
	if m.program().SubCaptureUncovered() {
		return nil, nil, nil, nil, fmt.Errorf("decoder.ForwardSubCapture: not wired for arch %q: a layer has an operation its contributions do not cover (Gemma 4's PLE branch or MoE block)", a.Name)
	}
	nL := a.NumLayers
	cache.c.SubCapture = true
	cache.c.SubAttn = make([][]float32, nL)
	cache.c.SubMLP = make([][]float32, nL)
	cache.c.SubMLPPre = make([][]float32, nL)
	cache.c.SubCtx = make([][]float32, nL)
	defer func() {
		cache.c.SubCapture = false
		cache.c.SubAttn, cache.c.SubMLP, cache.c.SubMLPPre, cache.c.SubCtx = nil, nil, nil, nil
	}()
	if _, ferr := m.forward(id, cache); ferr != nil {
		return nil, nil, nil, nil, ferr
	}
	return cache.c.SubAttn, cache.c.SubMLP, cache.c.SubCtx, cache.c.SubMLPPre, nil
}
