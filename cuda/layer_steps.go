//go:build cuda

package cuda

import (
	"fmt"
	"slices"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/program"
)

// cudaStep is one entry of CUDA's fusion table (OP4cu-b2, docs/tasks/task-forward-architecture-2026-10.md §4.9a): one of
// today's decode segments, emitting exactly its launches. A static step is a graph-capturable launch run (captureGraphs
// records it); a dynamic one binds per-token state (the position, the key count, a host readback).
type cudaStep uint8

const (
	stepQKV      cudaStep = iota + 1 // static: segA, the input norm through the QK-norm and v-norm
	stepAttnGap                      // dynamic: decodeAttnGap, rope, the KV store into the bound slot, attention by lane
	stepMLA                          // dynamic: segAMLA
	stepOutFFN                       // static: segB, the output projection and its residual, then the FFN up to the router readback
	stepDeltaNet                     // static: deltaNetMixer then segBFFN, one captured segment
	stepMamba                        // static: mambaMixer
	stepFFNOnly                      // static: nonGatedMLP
	stepTail                         // dynamic, then static: layerTail, the router readback and expert DMA, then segC
)

var cudaStepNames = [...]string{stepQKV: "qkv", stepAttnGap: "attnGap", stepMLA: "mla", stepOutFFN: "outFFN",
	stepDeltaNet: "deltaNet", stepMamba: "mamba", stepFFNOnly: "ffnOnly", stepTail: "tail"}

func (s cudaStep) String() string {
	if int(s) < len(cudaStepNames) && cudaStepNames[s] != "" {
		return cudaStepNames[s]
	}
	return fmt.Sprintf("cudaStep(%d)", uint8(s))
}

// cudaSublayerSteps is CUDA's fusion table: each layer sublayer pattern a CUDA resident runs (program.SublayersOf's kinds, in
// order, and whether they are a parallel block's two halves) and the segment steps that run it. PLE and the layer scale appear
// only after a sequential dense FFN, where segBFFN runs them; Gemma 4's MoE block scales inside segC; segB runs a parallel block.
var cudaSublayerSteps = []struct {
	pattern  []program.Sublayer
	parallel bool
	steps    []cudaStep
}{
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN}, false, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN}, true, attnSteps},                                                   // a parallel block
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN, program.SublayerLayerScale}, false, attnSteps},                      // Gemma 4's dense layer
	{[]program.Sublayer{program.SublayerAttention, program.SublayerDenseFFN, program.SublayerPLE, program.SublayerLayerScale}, false, attnSteps}, // Gemma 4's E-model layer
	{[]program.Sublayer{program.SublayerAttention, program.SublayerMoE}, false, attnSteps},
	{[]program.Sublayer{program.SublayerAttention, program.SublayerGemma4MoE}, false, attnSteps},
	{[]program.Sublayer{program.SublayerAttention}, false, attnSteps}, // Nemotron-H's attention block
	{[]program.Sublayer{program.SublayerMLA, program.SublayerDenseFFN}, false, mlaSteps},
	{[]program.Sublayer{program.SublayerMLA, program.SublayerMoE}, false, mlaSteps},
	{[]program.Sublayer{program.SublayerDeltaNet, program.SublayerDenseFFN}, false, deltaNetSteps},
	{[]program.Sublayer{program.SublayerDeltaNet, program.SublayerMoE}, false, deltaNetSteps},
	{[]program.Sublayer{program.SublayerMamba}, false, mambaSteps},         // Nemotron-H's Mamba-2 block
	{[]program.Sublayer{program.SublayerNonGatedFFN}, false, ffnOnlySteps}, // Nemotron-H's MLP block
}

var (
	attnSteps     = []cudaStep{stepQKV, stepAttnGap, stepOutFFN, stepTail}
	mlaSteps      = []cudaStep{stepMLA, stepOutFFN, stepTail}
	deltaNetSteps = []cudaStep{stepDeltaNet, stepTail}
	mambaSteps    = []cudaStep{stepMamba, stepTail}
	ffnOnlySteps  = []cudaStep{stepFFNOnly, stepTail}
)

// cudaSublayersOf is a layer's sublayer pattern, program.SublayersOf's kinds checked by program.CheckSublayerClaims, and whether
// they hold a parallel block.
func cudaSublayersOf(ops []program.Op) (pattern []program.Sublayer, parallel bool, err error) {
	spans, err := program.SublayersOf(ops, "CUDA")
	if err == nil {
		err = program.CheckSublayerClaims(ops, spans)
	}
	if err != nil {
		return nil, false, err
	}
	pattern = make([]program.Sublayer, len(spans))
	for i, sp := range spans {
		pattern[i] = sp.Kind
		parallel = parallel || sp.Parallel
	}
	return pattern, parallel, nil
}

// cudaStepsOf is a layer's steps from its operations: its sublayer pattern and parallel flag (cudaSublayersOf) looked up in
// cudaSublayerSteps. A layer whose operations match no sublayers, whose sublayers claim an operation their blocks do not emit, or
// whose pattern has no entry, is an error naming it.
func cudaStepsOf(ops []program.Op) ([]cudaStep, error) {
	pattern, parallel, err := cudaSublayersOf(ops)
	if err != nil {
		return nil, err
	}
	for _, e := range cudaSublayerSteps {
		if e.parallel == parallel && slices.Equal(e.pattern, pattern) {
			return slices.Clone(e.steps), nil
		}
	}
	return nil, fmt.Errorf("its sublayers %v (a parallel block: %v) have no entry in CUDA's fusion table", pattern, parallel)
}

// loadedSublayers is the sublayer pattern of what a built layer's upload holds, read from its buffers: the mixer's weights
// (DeltaNet's, Mamba-2's, MLA's, or attention's q), the FFN's (a router with Gemma 4's MoE norms and its dense branch's gate, or
// a router alone; an up projection without a gate, which is Nemotron-H's non-gated MLP; or a gate), then a PLE gate wherever
// one was uploaded, and a non-zero layer scalar on any layer but Gemma 4's MoE block, which scales inside segC. A Gemma 4 MoE layer
// without its dense gate holds no FFN the walk can run, so it reads as none. The buffers do not tell a gated dense FFN's
// activation, nor attention's form (plain, gated or Gemma 4's): the build checks those where it reads them.
func loadedSublayers(L *cudaLayer) []program.Sublayer {
	var s []program.Sublayer
	switch {
	case L.dnQKV.N > 0:
		s = append(s, program.SublayerDeltaNet)
	case L.mbIn.N > 0:
		s = append(s, program.SublayerMamba)
	case L.mlaQA.N > 0 || L.mlaQ.N > 0:
		s = append(s, program.SublayerMLA)
	case L.q.N > 0:
		s = append(s, program.SublayerAttention)
	}
	switch {
	case L.routerW != (Buffer{}) && L.g4preFFN2 != (Buffer{}):
		if L.g.N > 0 {
			s = append(s, program.SublayerGemma4MoE)
		}
	case L.routerW != (Buffer{}):
		s = append(s, program.SublayerMoE)
	case L.u.N > 0 && L.g.N == 0:
		s = append(s, program.SublayerNonGatedFFN)
	case L.g.N > 0:
		s = append(s, program.SublayerDenseFFN)
	}
	if L.pleGate.N > 0 {
		s = append(s, program.SublayerPLE)
	}
	if L.layerScalar != 0 && L.g4preFFN2 == (Buffer{}) { // any layer but Gemma 4's MoE block, which scales inside segC
		s = append(s, program.SublayerLayerScale)
	}
	return s
}

// checkLoadedSublayers is an error unless a layer's program sublayers and parallel flag are what its upload holds.
func checkLoadedSublayers(prog []program.Sublayer, progParallel bool, loaded []program.Sublayer, loadedParallel bool) error {
	if !slices.Equal(prog, loaded) || progParallel != loadedParallel {
		return fmt.Errorf("the program's sublayers are %v (a parallel block: %v) and the layer's upload holds %v (parallel: %v)",
			prog, progParallel, loaded, loadedParallel)
	}
	return nil
}

// walkPlant plants one defect in the walk, so B2-2 can show B2-1 goes red; zero in production (set by tests only, through
// setWalkPlantForTest). The plants act in runStep's live path, so they are run with CUDA graphs off: a captured segment replays
// without passing them.
var walkPlant walkPlantKind

type walkPlantKind uint8

const (
	walkPlantNone       walkPlantKind = iota
	walkPlantDropGap                  // the walk skips attnGap
	walkPlantSwap                     // the walk runs outFFN before attnGap
	walkPlantQDimMinus1               // outFFN reads the layer's qDim as qDim-1
)

// legacyLayerForTest, when set, runs a layer with the loop launchToken had before the walk, for B2-1's comparison and B2-6's
// A/B. A build made while it is set also skips the walk's build checks, as the pre-walk build did. Nil in production; tests set
// it directly and restore it.
var legacyLayerForTest func(r *cudaResident, Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error

// walkLayer runs layer l's steps (OP4cu-b2). gA, gB and gC say whether the static segments replay their captured graphs.
func (r *cudaResident) walkLayer(Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error {
	steps := Ly.steps
	if walkPlant == walkPlantSwap && len(steps) == 4 {
		steps = []cudaStep{steps[0], steps[2], steps[1], steps[3]}
	}
	for _, st := range steps {
		if walkPlant == walkPlantDropGap && st == stepAttnGap {
			continue
		}
		if e := r.runStep(st, Ly, l, pos, ropePos, qTempScale, gA, gB, gC); e != nil {
			return e
		}
	}
	return nil
}

// runStep emits one step's launches, replaying a static step's captured graph when graphs are on.
func (r *cudaResident) runStep(st cudaStep, Ly *cudaLayer, l, pos, ropePos int, qTempScale float32, gA, gB, gC bool) error {
	switch st {
	case stepQKV:
		if gA {
			return r.replay(Ly.gSegA)
		}
		return r.segA(Ly, l)
	case stepAttnGap:
		return r.decodeAttnGap(Ly, l, pos, ropePos, qTempScale)
	case stepMLA:
		return r.segAMLA(Ly, l, pos, ropePos)
	case stepOutFFN:
		if r.subCap { // pre-o-proj attention context (qDim), before quant — the cross-box discriminator (live path only)
			r.capVec(r.cctx, r.subCtxC, l, Ly.qDim)
		}
		if gB {
			return r.replay(Ly.gSegB)
		}
		if walkPlant == walkPlantQDimMinus1 {
			p := *Ly
			p.qDim--
			return r.segB(&p, l, r.x)
		}
		return r.segB(Ly, l, r.x)
	case stepDeltaNet:
		// The mixer and its FFN half are captured as ONE segment (gSegA): with no rope or attention gap there is nothing
		// dynamic to split around.
		if gA {
			return r.replay(Ly.gSegA)
		}
		if e := r.deltaNetMixer(Ly, l); e != nil {
			return e
		}
		return r.segBFFN(Ly, l, r.x)
	case stepMamba:
		return r.mambaMixer(Ly)
	case stepFFNOnly:
		return r.nonGatedMLP(Ly)
	case stepTail:
		return r.layerTail(Ly, l, gC, r.x)
	}
	return fmt.Errorf("cuda: layer %d: step %v has no launches", l, st)
}

// replay runs a captured segment, then syncs the stream when graphsSync asks for it.
func (r *cudaResident) replay(g *gpu.Graph) error {
	if e := g.Replay(); e != nil {
		return e
	}
	if r.graphsSync {
		return r.stream.Sync()
	}
	return nil
}

// prefillStep is one block of the batched prefill's layer body (prefillCore, OP4cu-d1, docs/tasks/task-forward-architecture-2026-10.md
// §4.9a): the DeltaNet rows or the attention half, then the MoE or the dense FFN. Each runs over a pass's M rows.
type prefillStep uint8

const (
	pstepDeltaNet prefillStep = iota + 1 // prefillDeltaNetRows: the Gated-DeltaNet mixer, its projections batched and the recurrence row by row
	pstepAttn                            // q/k/v, rope and attention by lane (the step's rows through decode's gap), the output projection
	pstepMoE                             // the MoE FFN, Gemma 4's block included: per row off the batched residual, or expert-major
	pstepDenseFFN                        // the dense FFN, with the PLE branch and the layer scale
)

func (s prefillStep) String() string {
	switch s {
	case pstepDeltaNet:
		return "DeltaNet rows"
	case pstepAttn:
		return "attention"
	case pstepMoE:
		return "MoE FFN"
	case pstepDenseFFN:
		return "dense FFN"
	}
	return fmt.Sprintf("prefillStep(%d)", uint8(s))
}

// cudaPrefillStepsOf is a layer's batched prefill steps from its operations: its mixer's block, then its FFN's. A layer whose
// sublayers the batched pass has no blocks for (Mamba-2, MLA, an FFN-only or an attention-only block) has none, and
// prefillStaticDecline sends its model to the sequential path.
func cudaPrefillStepsOf(ops []program.Op) []prefillStep {
	pattern, _, err := cudaSublayersOf(ops)
	if err != nil || len(pattern) < 2 {
		return nil
	}
	// Past the mixer and the FFN, a layer may carry only what the dense FFN block runs after a dense FFN, in its order: the PLE
	// branch, then the layer scale, each at most once (cudaSublayerSteps' Gemma 4 rows). Anything else has no batched block.
	if tail := pattern[2:]; len(tail) > 0 {
		if pattern[1] != program.SublayerDenseFFN {
			return nil
		}
		if len(tail) > 0 && tail[0] == program.SublayerPLE {
			tail = tail[1:]
		}
		if len(tail) > 0 && tail[0] == program.SublayerLayerScale {
			tail = tail[1:]
		}
		if len(tail) > 0 {
			return nil
		}
	}
	var steps []prefillStep
	switch pattern[0] {
	case program.SublayerAttention:
		steps = append(steps, pstepAttn)
	case program.SublayerDeltaNet:
		steps = append(steps, pstepDeltaNet)
	default:
		return nil
	}
	switch pattern[1] {
	case program.SublayerMoE, program.SublayerGemma4MoE:
		return append(steps, pstepMoE)
	case program.SublayerDenseFFN:
		return append(steps, pstepDenseFFN)
	}
	return nil
}

// prefillStepsFromFlags is the steps the batched prefill's former branches chose for a built layer: the DeltaNet rows or the
// attention half, then the MoE or the dense FFN. The build checks the program's steps against it.
func prefillStepsFromFlags(L *cudaLayer) []prefillStep {
	steps := []prefillStep{pstepAttn, pstepDenseFFN}
	if L.isDeltaNet() {
		steps[0] = pstepDeltaNet
	}
	if L.g4moe || L.isMoE {
		steps[1] = pstepMoE
	}
	return steps
}

// prefillPlantKind is one defect planted in the batched prefill's walk, so OP4cu-d1's D1-2 can show it goes red. prefillPlant is
// a constant prefillPlantNone outside goinfer_testhooks builds (prefill_plant_off.go), so production compiles the plants away.
type prefillPlantKind uint8

const (
	prefillPlantNone         prefillPlantKind = iota
	prefillPlantDropDense                     // the walk skips the dense FFN step
	prefillPlantSwap                          // the walk runs the FFN step before the mixer's
	prefillPlantDropMoE                       // the walk skips the MoE step
	prefillPlantDropDeltaNet                  // the walk skips the DeltaNet step
)

// prefillPlantDrops is the step each drop plant removes.
var prefillPlantDrops = map[prefillPlantKind]prefillStep{
	prefillPlantDropDense: pstepDenseFFN, prefillPlantDropMoE: pstepMoE, prefillPlantDropDeltaNet: pstepDeltaNet,
}

// prefillLayerSteps is the steps prefillCore runs for layer L.
func (r *cudaResident) prefillLayerSteps(L *cudaLayer) []prefillStep {
	steps := L.prefillSteps
	if prefillPlant == prefillPlantNone {
		return steps
	}
	if drop, ok := prefillPlantDrops[prefillPlant]; ok {
		return slices.DeleteFunc(slices.Clone(steps), func(s prefillStep) bool { return s == drop })
	}
	if prefillPlant == prefillPlantSwap && len(steps) == 2 {
		return []prefillStep{steps[1], steps[0]}
	}
	return steps
}
